#!/usr/bin/env python3
"""One command: validate -> prepare -> all registered runs -> aggregate.
Run independent jobs on separate GPUs; resource-aware queue, automatic resume.
"""
import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from project_config import load_config

HERE = Path(__file__).resolve().parent
GIB = 1024**3
ALL_METHODS = ['continue', 'random_copy', 'traffic_copy', 'eu_gn', 'graftmoe', 'graftmoe_native',
               'cluster_no_warm', 'random_groups_warm']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.building')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parameters(m):
    c = read(Path(m['path'])/'config.json')
    d, L, N = c['hidden_size'], c['num_hidden_layers'], m['old_experts']
    f = c.get('moe_intermediate_size', c['intermediate_size'])
    expert = 3*d*f
    shared = L*(3*d*c.get('shared_expert_intermediate_size', 0) + (d if c.get('shared_expert_intermediate_size', 0) else 0))
    kv = c.get('num_key_value_heads', c['num_attention_heads']) * (d//c['num_attention_heads'])
    embedding = c['vocab_size']*d*(1 if c.get('tie_word_embeddings', False) else 2)
    base = L*N*expert + shared + L*(2*d*d+2*d*kv+N*d+2*d) + embedding
    extra = L*(m['expand_to']-N)*expert
    return base, extra, L, expert


def requirements(cfg, key, method):
    m = cfg['models'][key]
    base, extra, layers, expert = parameters(m)
    load_cpu = 2*base/GIB + 8
    if method == 'prepare_gn':
        train = layers*m['old_experts']*expert
        return {'gpu_gib': (2*base+2*train)/GIB+12,
                'cpu_gib': max(load_cpu, 4*train/GIB+16), 'disk_gib': 1.}
    if method == 'original':
        return {'gpu_gib': 2*base/GIB+8, 'cpu_gib': load_cpu, 'disk_gib': 2.}
    expanded = method != 'continue'
    train = extra + layers*m['expand_to']*read(Path(m['path'])/'config.json')['hidden_size'] if expanded else base
    total = base + (extra if expanded else 0)
    warm = 2*extra if method in ('graftmoe', 'graftmoe_native', 'random_groups_warm') else 0
    retain = 2*(extra if expanded else total) if cfg['save_final_weights'] else 0
    return {'gpu_gib': (2*total+2*train)/GIB+12,
            'cpu_gib': max(load_cpu, (4*train+4*train/layers)/GIB+16),
            'disk_gib': (2.08*4*train+warm+retain)/GIB+5}


def available_gpus():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(line.split(',')[0]): float(line.split(',')[1])/1024 for line in text.strip().splitlines()}


def validate(cfg, keys):
    # Resolve exact inputs once; no glob("latest") or silent path substitutions.
    import baseline_utils as bu
    for key in keys:
        m = cfg['models'][key]
        c = read(Path(m['path'])/'config.json')
        expected = 'olmoe' if key == 'olmoe' else 'qwen2_moe'
        if c['model_type'] != expected or c['num_experts'] != m['old_experts'] or c['num_experts_per_tok'] != m['top_k']:
            raise ValueError(f'{key}: native configuration differs from registered protocol.')
        dr = read(Path(m['descriptors'])/'report.json')
        if not dr.get('complete') or dr['arguments']['model'] != key:
            raise ValueError(f'{key}: descriptors are not a complete matching-model collection.')
        actual = bu.checkpoint_identity(Path(m['path']))
        for item in ('config_sha256', 'index_sha256', 'weight_files'):
            if actual[item] != dr['checkpoint'][item]:
                raise ValueError(f'{key}: checkpoint identity differs from descriptor source ({item}).')
        for name, sha in dr['files_sha256'].items():
            if bu.sha256(Path(m['descriptors'])/name) != sha:
                raise ValueError(f'{key}: descriptor checksum failed: {name}')
        for split in ('train', 'valid', 'test', 'calibration'):
            a = SimpleNamespace(data_dir=Path(m['data']), split=split, model=key,
                                model_path=Path(m['path']), max_sequences=0)
            data, count, info = bu.validate_data(a)
            if split == 'calibration' and (info['sha256'] != dr['data']['sha256'] or
                    info['preparation_report_sha256'] != dr['data']['preparation_report_sha256']):
                raise ValueError(f'{key}: descriptor calibration corpus differs from prepared DCLM data.')
            if data.shape[1] != 2048:
                raise ValueError('Protocol requires packed length 2048.')
            if split == 'train' and cfg['steps']*cfg['accumulation'] > len(data):
                raise ValueError('Training budget exceeds one pass of prepared train blocks.')
        print(f'Inputs verified: {key}; data/tokenizer/checkpoint/descriptors.', flush=True)
    for name in ('transformers', 'accelerate', 'psutil', 'numpy', 'matplotlib'):
        print(f'{name}: {importlib.metadata.version(name)}', flush=True)
    if importlib.metadata.version('transformers') != '4.57.1':
        raise ValueError('This release requires transformers==4.57.1 for its native expert implementation.')
    if cfg['tasks'] and importlib.metadata.version('lm_eval') != '0.4.9.1':
        raise ValueError('Use lm-eval==0.4.9.1 for frozen task definitions; see requirements.txt.')
    if cfg['steps'] < 1 or cfg['accumulation'] < 1 or cfg['eval_every'] < 1 or cfg['save_every'] < 1:
        raise ValueError('Invalid training or checkpoint interval.')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Training seeds must be nonempty and distinct.')
    if any(not isinstance(seed, int) or seed < 0 for seed in cfg['seeds']):
        raise ValueError('Training seeds must be nonnegative integers.')


def prepare_groups(cfg, key, root):
    import group_utils
    target = root/'prepared'/key/'groups'
    if (target/'report.json').exists():
        report = read(target/'report.json')
        source = Path(cfg['models'][key]['descriptors'])/'report.json'
        if not report.get('complete') or report.get('source_report_sha256') != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError('Prepared groups differ from the current descriptor source; use a new experiment root.')
        for name, checksum in report['plan_files_sha256'].items():
            if hashlib.sha256((target/'plans'/name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f'Prepared expansion plan changed: {name}')
        return
    building = target.with_name('groups.building')
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    args = SimpleNamespace(descriptor_dir=Path(cfg['models'][key]['descriptors']),
                           expand_to=cfg['models'][key]['expand_to'], clusters=cfg['clusters'],
                           seeds=cfg['seeds'] if len(cfg['seeds'])>1 else [cfg['seeds'][0], cfg['seeds'][0]+1],
                           n_init=10, max_iter=300, random_groups=100)
    group_utils.run(args, building)
    os.replace(building, target)


def get_plan(cfg, key, method, seed, root):
    if method in ('continue', 'original'):
        return None
    names = {'graftmoe': 'full', 'graftmoe_native': 'full', 'cluster_no_warm': 'full', 'random_groups_warm': 'random_groups',
             'random_copy': 'random_copy', 'traffic_copy': 'traffic_top', 'eu_gn': 'full'}
    selected_seed = cfg['seeds'][0] if method == 'traffic_copy' else seed
    plan = read(root/'prepared'/key/'groups'/'plans'/f'{names[method]}_seed{selected_seed}.json')
    plan['training_seed'] = seed
    if method == 'eu_gn':
        report = read(root/'prepared'/key/'gradient_scores.json')
        bylayer = {r['layer']: r['scores'] for r in report['layers']}
        for row in plan['layers']:
            values = bylayer[row['layer']]
            order = sorted(range(len(values)), key=lambda e: (-values[e], e))
            # Explicit definition: max THREE ADDED copies of any source expert.
            parents = [e for e in order for _ in range(3)][:plan['new_experts_per_layer']]
            if len(parents) != plan['new_experts_per_layer']:
                raise ValueError('Growth exceeds EU-GN cap-three capacity.')
            row['parents'], row['labels'], row['groups'] = parents, None, None
            row['utility_scores'] = values
        plan['method'] = 'eu_gn_adapted'
        plan['copy_rule'] = 'Squared mean gradient greedy; cap=3 added copies; new router bias U(-1e-3,1e-3).'
        plan['adaptation'] = report['adaptation']
        plan['gradient_preparation_seconds'] = report['seconds']
    return plan


def job_path(root, key, method, seed):
    return root/'runs'/f'{key}_{method}_seed{seed}'


def launch_queue(cfg, keys, jobs, root, gpus, parallel):
    import psutil
    pending = list(jobs)
    running = []
    failures = []
    def stop_workers():
        # These are only the worker processes created by this scheduler.
        for item in running:
            process = item['process']
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for item in running:
            try:
                item['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(item['process'].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            item['log'].close()
    atexit.register(stop_workers)
    logdir = root/'logs'
    logdir.mkdir(exist_ok=True)
    wait_started = time.monotonic()
    last_notice = 0
    while pending or running:
        for item in list(running):
            status = item['process'].poll()
            if status is not None:
                item['log'].close()
                running.remove(item)
                if status:
                    failures.append({'job': item['job'], 'status': status, 'log': str(item['path'])})
                    print(f'FAILED: {item["job"]}; inspect {item["path"]}', flush=True)
                else:
                    print(f'Finished: {item["job"]}', flush=True)
                wait_started = time.monotonic()
        free = available_gpus()
        started = False
        for job in list(pending):
            if len(running) >= parallel:
                break
            key, method, seed = job
            complete = (root/'prepared'/key/'gradient_scores.json') if method == 'prepare_gn' else (job_path(root,*job)/'complete.json')
            if complete.exists():
                if method != 'prepare_gn' and not cfg['keep_optimizer_checkpoints']:
                    from graftmoe_core import cleanup_completed_run
                    cleanup_completed_run(complete.parent)
                pending.remove(job)
                continue
            r = requirements(cfg, key, method)
            reserved_disk = sum(x['requirements']['disk_gib'] for x in running)
            # Conservative full remaining reservations, even if some states already written.
            reserved_cpu = sum(x['requirements']['cpu_gib'] for x in running)
            disk_ok = shutil.disk_usage(root).free/GIB >= r['disk_gib'] + reserved_disk + cfg['disk_reserve_gib']
            ram_ok = psutil.virtual_memory().available/GIB >= r['cpu_gib'] + reserved_cpu
            chosen = next((g for g in gpus if g not in [x['gpu'] for x in running]
                           and free.get(g, 0) >= r['gpu_gib']), None)
            if not (disk_ok and ram_ok and chosen is not None):
                continue
            path = logdir/f'{key}_{method}_seed{seed}.log'
            log = path.open('a', buffering=1)
            log.write(f'\n--- invocation {time.strftime("%Y-%m-%d %H:%M:%S")} ---\n')
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(chosen)
            env['TOKENIZERS_PARALLELISM'] = 'false'
            env['OMP_NUM_THREADS'] = str(cfg['cpu_threads'])
            if cfg['tasks']:
                env['HF_HUB_OFFLINE'] = '1'
                env['HF_DATASETS_OFFLINE'] = '1'
            command = [sys.executable, '-u', str(HERE/'run_graftmoe.py'), '--worker', method,
                       '--model', key, '--seed', str(seed), '--root', str(root)]
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, cwd=HERE, start_new_session=True)
            write(logdir/f'{key}_{method}_seed{seed}.process.json',
                  {'pid': p.pid, 'create_time': psutil.Process(p.pid).create_time(),
                   'gpu': chosen, 'job': list(job), 'command': command})
            running.append({'job': job, 'process': p, 'gpu': chosen, 'log': log, 'path': path, 'requirements': r})
            pending.remove(job)
            started = True
            wait_started = time.monotonic()
            print(f'Launched GPU {chosen}: {job}; log={path}', flush=True)
        if pending and not started and time.monotonic()-last_notice > 60:
            print(f'Queue: running={len(running)}, pending={len(pending)}, '
                  f'disk_free={shutil.disk_usage(root).free/GIB:.1f} GiB, '
                  f'RAM_available={psutil.virtual_memory().available/GIB:.1f} GiB. Resource details: '
                  f'{[(j,requirements(cfg,j[0],j[1])) for j in pending[:2]]}', flush=True)
            last_notice = time.monotonic()
        if pending and not running and time.monotonic()-wait_started > cfg['resource_wait_minutes']*60:
            raise RuntimeError('No job fits current GPU/RAM/disk resources. Inputs/results preserved. '
                               'Use a larger output filesystem or free resources, then repeat same command.')
        if pending or running:
            time.sleep(5)
    write(root/'last_queue_failures.json', failures)
    atexit.unregister(stop_workers)
    if failures:
        raise RuntimeError('Some jobs failed. Fix the cause shown in logs and repeat the same command to resume.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--models', nargs='+', choices=['olmoe','qwen'], default=['olmoe','qwen'])
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS+['original'])
    parser.add_argument('--seeds', nargs='+', type=int, help='Override the configured training seeds for this root.')
    parser.add_argument('--gpus', nargs='+', type=int, default=[4,5])
    parser.add_argument('--max-parallel', type=int, default=1)
    parser.add_argument('--check', action='store_true', help='Read-only provenance / resource check; no training.')
    parser.add_argument('--worker', choices=ALL_METHODS+['original','prepare_gn'], help=argparse.SUPPRESS)
    parser.add_argument('--model', choices=['olmoe','qwen'], help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    args.methods = list(dict.fromkeys(args.methods or ALL_METHODS))
    args.methods = [m for m in args.methods if m != 'original']
    if args.worker:
        import graftmoe_core as core
        root = args.root.resolve()
        cfg = read(root/'protocol_frozen.json')
        if args.worker == 'prepare_gn':
            core.gradient_scores(cfg, args.model, 'cuda:0', root/'prepared'/args.model)
        else:
            plan = get_plan(cfg, args.model, args.worker, args.seed, root)
            core.run_experiment(cfg, args.model, args.worker, args.seed,
                                job_path(root,args.model,args.worker,args.seed), plan, 'cuda:0')
        return
    cfg = load_config(args.config)
    if args.seeds is not None:
        cfg['seeds'] = args.seeds
    root = args.root or Path(cfg['output_root'])
    root = root.expanduser().resolve()
    if args.max_parallel < 1 or len(set(args.gpus)) != len(args.gpus):
        parser.error('max-parallel must be positive; GPU indices must be distinct.')
    validate(cfg, args.models)
    print('Training matrix: original once/model; selected methods x models x seeds.', flush=True)
    for key in args.models:
        for method in args.methods:
            print(key, method, requirements(cfg, key, method), flush=True)
    if args.check:
        print('Read-only check passed. No training started.')
        return
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    root_lock = (root/'.scheduler.lock').open('a')
    try:
        fcntl.flock(root_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another scheduler is using {root}.')
    frozen = root/'protocol_frozen.json'
    if frozen.exists() and read(frozen) != cfg:
        raise ValueError('Protocol differs from existing root; choose --root for a new experiment.')
    write(frozen, cfg)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}
    if (root/'run_sources.json').exists() and read(root/'run_sources.json') != sources:
        raise ValueError('Code changed since this experiment root was created. Preserve the original version for exact resume.')
    write(root/'run_sources.json', sources)
    selection = {'models': args.models, 'methods': args.methods, 'seeds': cfg['seeds'],
                 'gpus': args.gpus, 'max_parallel': args.max_parallel}
    previous_selection = root/'requested_matrix.json'
    if previous_selection.exists():
        previous = read(previous_selection)
        if any(previous[field] != selection[field] for field in ('models', 'methods', 'seeds')):
            raise ValueError('Requested matrix differs from this root. Use a separate --root for a different selection.')
    write(root/'requested_matrix.json', selection)
    if cfg['tasks']:
        ready = Path(cfg['benchmark_ready_root'])
        marker = ready/'offline_ready.json'
        manifest = ready/'benchmark_data.json'
        if not marker.exists() or not manifest.exists():
            raise RuntimeError('Run prepare_eval_data.py download, then prepare_eval_data.py check first.')
        status = read(marker)
        if (not status.get('complete') or status.get('tasks') != cfg['tasks'] or
            status.get('manifest_sha256') != hashlib.sha256(manifest.read_bytes()).hexdigest()):
            raise RuntimeError('Evaluation readiness mismatch; rerun prepare_eval_data.py check.')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        target = root/'benchmark_data.json'
        if target.exists() and read(target) != read(manifest):
            raise ValueError('Evaluation data changed within existing experiment.')
        write(target, read(manifest))
        from task_data import prepare
        prepare(cfg['tasks'], root)
    for key in args.models:
        prepare_groups(cfg, key, root)
    if 'eu_gn' in args.methods:
        launch_queue(cfg, args.models, [(k,'prepare_gn',cfg['seeds'][0]) for k in args.models],
                     root, args.gpus, min(1,args.max_parallel))
    preparation = {}
    for key in args.models:
        source = read(Path(cfg['models'][key]['descriptors'])/'report.json')
        grouping = read(root/'prepared'/key/'groups'/'report.json')
        gradient_path = root/'prepared'/key/'gradient_scores.json'
        preparation[key] = {'descriptor_calibration_seconds': source['calibration_seconds'],
                            'descriptor_load_seconds': source['load_seconds'],
                            'descriptor_probe_and_save_seconds': source['probe_and_save_seconds'],
                            'all_candidate_grouping_cpu_seconds': grouping['elapsed_seconds'],
                            'gradient_scoring_seconds': read(gradient_path)['seconds'] if gradient_path.exists() else 0,
                            'note': 'Shared per-model preparation; grouping timing includes all candidates/seeds, '
                                    'not a separately measured per-method cost.'}
    write(root/'preparation_costs.json',preparation)
    jobs = [(k,'original',0) for k in args.models]
    for seed in cfg['seeds']:
        for method in args.methods:
            for key in args.models:
                jobs.append((key,method,seed))
    launch_queue(cfg, args.models, jobs, root, args.gpus, args.max_parallel)
    subprocess.run([sys.executable, str(HERE/'summarize_graftmoe.py'), '--root', str(root)], check=True)
    print(f'All requested runs finished. Tables/curves: {root}/summary', flush=True)


if __name__ == '__main__':
    main()
