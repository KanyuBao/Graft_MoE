#!/usr/bin/env python3
"""Re-evaluate a saved GRAFT-MoE run with native routing and no exploration.

By default, evaluate the entire frozen validation split and record NLL, PPL,
per-block losses, and expert-assignment counts. Use --split test for the held-out
test split. --tasks alone adds all frozen downstream tasks; names select a subset.
Every invocation writes to a new directory and leaves training results unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import uuid

from load_graftmoe_checkpoint import inspect_run, load_run


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True, help='Completed run with retained final weights.')
    parser.add_argument('--device', default='cuda:0', help='Device after CUDA_VISIBLE_DEVICES mapping, or cpu.')
    parser.add_argument('--split', choices=('valid', 'test'), default='valid')
    parser.add_argument('--max-blocks', type=int, default=0, help='0: complete split; positive: first N packed blocks, labeled as a subset.')
    parser.add_argument('--tasks', nargs='*', default=None, metavar='TASK', help='Also evaluate all frozen tasks, or the listed subset; omitted: language modeling only.')
    parser.add_argument('--benchmark-root', type=Path, help='Root containing task readiness records; defaults to the original experiment root.')
    parser.add_argument('--output', type=Path, help='New result directory; an existing directory is never overwritten.')
    parser.add_argument('--check', action='store_true', help='Check run metadata and held-out file presence without loading model tensors.')
    args = parser.parse_args()
    if args.max_blocks < 0:
        parser.error('--max-blocks must be nonnegative')
    info = inspect_run(args.run_dir)
    run, recipe = info['run'], info['recipe']
    cfg, key = recipe['config'], recipe['model']
    task_names = [] if args.tasks is None else (args.tasks or cfg['tasks'])
    task_names = list(dict.fromkeys(task_names))
    unknown = set(task_names)-set(cfg['tasks'])
    if unknown:
        parser.error(f'Tasks are not part of this frozen protocol: {sorted(unknown)}')
    data_dir = Path(cfg['models'][key]['data'])
    data_path, report_path = data_dir/f'{args.split}.npy', data_dir/'report.json'
    if not data_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f'The recorded held-out split and report.json are required in {data_dir}')
    if _sha256(report_path) != recipe['data_report_sha256']:
        raise ValueError('The data preparation report differs from the frozen training recipe.')
    benchmark_root = args.benchmark_root.expanduser().resolve() if args.benchmark_root else run.parent.parent
    if args.check:
        print(f"Evaluation inputs present: {key}/{recipe['method']}; split={args.split}; tasks={task_names}")
        print('Tensor/data checksums, model loading, and downstream task readiness are verified when evaluation runs.')
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import numpy as np
    import torch
    import baseline_utils as bu
    import graftmoe_core as c

    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Use the configured GPU environment, or --device cpu if sufficient RAM is available.')
    c.seed_all(recipe['seed'])
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    data = c.data_for(cfg, key, args.split)
    count = min(args.max_blocks, len(data)) if args.max_blocks else len(data)
    indices = np.arange(count, dtype=np.int64)
    output = args.output.expanduser().resolve() if args.output else run/'evaluations'/(
        args.split+'_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new directory.')
    if task_names:
        if not (benchmark_root/'benchmark_data.json').is_file():
            raise FileNotFoundError(f'Downstream readiness record is missing: {benchmark_root / "benchmark_data.json"}')
    model, _ = load_run(run, args.device)
    output.mkdir(parents=True, exist_ok=False)
    lm = c.evaluate(model, data, indices, output, args.split, cfg['models'][key]['old_experts'],
                    args.device, step=info['complete']['final_step'])
    tasks = c.run_tasks(model, key, cfg, output, args.device, benchmark_root=benchmark_root,
                        task_names=task_names) if task_names else {}
    result = {'complete': True, 'run_dir': str(run), 'model': key, 'method': recipe['method'],
              'seed': recipe['seed'], 'checkpoint_kind': info['kind'], 'device': args.device,
              'final_step': info['complete']['final_step'], 'split': args.split,
              'full_split': count==len(data), 'blocks': count, 'available_blocks': len(data),
              'selection': 'first N packed blocks' if count < len(data) else 'all packed blocks',
              'routing': 'native inference routing; exploration disabled',
              'language_modeling': lm, 'tasks': tasks,
              'recipe_sha256': _sha256(run/'recipe.json'), 'evaluator_sha256': _sha256(__file__)}
    bu.save_json(output/'evaluation.json', result)
    print(f"Evaluation saved: {output/'evaluation.json'}; full_split={count==len(data)}")
    if count < len(data):
        print('Subset evaluation: omit --max-blocks to report the complete frozen split.')


if __name__ == '__main__':
    main()
