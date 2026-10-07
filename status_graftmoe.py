#!/usr/bin/env python3
"""Show the recorded progress of every requested experiment without loading models."""
import argparse
from collections import deque
import json
from pathlib import Path


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def last_train_step(path):
    if not path.exists():
        return 0
    with path.open(encoding='utf-8') as handle:
        tail = deque(handle, maxlen=3)
    for line in reversed(tail):
        try:
            return int(json.loads(line)['step'])
        except (ValueError, KeyError, TypeError):
            continue
    return 0


def process_status(path):
    record = read(path)
    if not record:
        return 'unknown'
    try:
        import psutil
        proc = psutil.Process(record['pid'])
        if abs(proc.create_time() - record['create_time']) > 0.01:
            return 'stopped'
        return 'running' if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE else 'stopped'
    except ImportError:
        return 'unknown'
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return 'stopped'
    except psutil.AccessDenied:
        return 'unknown'


def collect(root):
    cfg = read(root / 'protocol_frozen.json')
    matrix = read(root / 'requested_matrix.json')
    if cfg is None or matrix is None:
        raise FileNotFoundError('The experiment root has no frozen protocol/requested matrix yet.')
    jobs = [(model, 'original', 0) for model in matrix['models']]
    jobs += [(model, method, seed) for model in matrix['models']
             for method in matrix['methods'] for seed in matrix['seeds']]
    rows = []
    for model, method, seed in jobs:
        name = f'{model}_{method}_seed{seed}'
        run = root / 'runs' / name
        report = read(run / 'complete.json')
        latest = read(run / 'latest.json', {})
        total = 0 if method == 'original' else cfg['steps']
        step = report['final_step'] if report else max(last_train_step(run / 'train.jsonl'), latest.get('step', 0))
        tasks = len(report['tasks']) if report else sum((run / f'task_{task}.json').exists() for task in cfg['tasks'])
        process = process_status(root / 'logs' / f'{name}.process.json')
        if report and report.get('complete'):
            stage = 'complete'
        elif (run / 'test_final.json').exists():
            stage = 'tasks/finalization'
        elif step >= total and method != 'original':
            stage = 'final evaluation'
        elif step:
            stage = 'training'
        elif (run / 'warm_cache').exists() or (run / 'warm_report.json').exists():
            stage = 'warm-start'
        elif (run / 'recipe.json').exists():
            stage = 'initialization/evaluation'
        else:
            stage = 'pending'
        final = report['test'] if report else read(run / 'test_final.json', {})
        rows.append({'run': name, 'step': step, 'total_steps': total,
                     'checkpoint_step': latest.get('step'), 'tasks_done': tasks,
                     'tasks_total': len(cfg['tasks']), 'test_ppl': final.get('ppl'),
                     'stage': stage, 'process': process})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--json', action='store_true', help='Print machine-readable progress.')
    args = parser.parse_args()
    rows = collect(args.root.expanduser().resolve())
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    print(f"{'Run':42} {'Steps':11} {'Saved':7} {'Tasks':7} {'Test PPL':10} {'Process':9} Stage")
    for row in rows:
        steps = f"{row['step']}/{row['total_steps']}"
        tasks = f"{row['tasks_done']}/{row['tasks_total']}"
        ppl = f"{row['test_ppl']:.4f}" if row['test_ppl'] is not None else '--'
        saved = str(row['checkpoint_step']) if row['checkpoint_step'] is not None else '--'
        print(f"{row['run']:42} {steps:11} {saved:7} {tasks:7} {ppl:10} {row['process']:9} {row['stage']}")
    print(f"Completed: {sum(r['stage'] == 'complete' for r in rows)}/{len(rows)}")
    print('Steps are recorded progress; Saved is the resumable checkpoint. Process state is checked on this host.')


if __name__ == '__main__':
    main()
