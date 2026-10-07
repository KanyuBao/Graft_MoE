#!/usr/bin/env python3
"""Prepare evaluation data before training. No model loading, no GPU allocation.
Each benchmark is a separate child process: explicit progress, timeout, resumable cache.
"""
import argparse
import hashlib
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from project_config import load_config

HERE=Path(__file__).resolve().parent


def write(path,value):
    temporary=path.with_name(path.name+'.building')
    with temporary.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['download','check'])
    p.add_argument('--config',type=Path,default=HERE/'protocol.json')
    p.add_argument('--root',type=Path)
    p.add_argument('--tasks',nargs='+')
    p.add_argument('--timeout',type=int,default=900,help='Per task timeout in seconds.')
    p.add_argument('--refresh',action='store_true',help='Recheck/download tasks even when their manifests exist.')
    p.add_argument('--worker',help=argparse.SUPPRESS)
    args=p.parse_args()
    cfg=load_config(args.config)
    root=args.root or Path(cfg['benchmark_ready_root'])
    root.mkdir(parents=True,exist_ok=True)
    tasks=args.tasks or cfg['tasks']
    if any(t not in cfg['tasks'] for t in tasks):
        p.error('Task is not in this protocol.')
    if args.worker:
        faulthandler.enable()
        faulthandler.dump_traceback_later(60,repeat=True)
        from task_data import snapshots
        report=snapshots([args.worker])
        path=root/f'{args.worker}.json'
        if args.action=='check':
            if not path.exists() or json.loads(path.read_text())!=report:
                raise ValueError(f'Offline fingerprint mismatch: {args.worker}')
        else:
            write(path,report)
        faulthandler.cancel_dump_traceback_later()
        return
    (root/'offline_ready.json').unlink(missing_ok=True)
    for task in tasks:
        if args.action=='download' and not args.refresh and (root/f'{task}.json').exists():
            print(f'[ready] {task}: manifest present; download skipped (check will verify cache)',flush=True)
            continue
        print(f'[{args.action}] {task}; timeout={args.timeout}s; no GPU needed',flush=True)
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=''
        env['TOKENIZERS_PARALLELISM']='false'
        env['HF_HUB_ETAG_TIMEOUT']='20'
        env['HF_HUB_DOWNLOAD_TIMEOUT']='60'
        if args.action=='check':
            env['HF_HUB_OFFLINE']='1'; env['HF_DATASETS_OFFLINE']='1'
        else:
            env.pop('HF_HUB_OFFLINE',None); env.pop('HF_DATASETS_OFFLINE',None)
        command=[sys.executable,'-u',str(HERE/'prepare_eval_data.py'),args.action,
                 '--config',str(args.config.resolve()),'--root',str(root.resolve()),'--worker',task]
        try:
            subprocess.run(command,env=env,check=True,timeout=args.timeout)
        except subprocess.TimeoutExpired:
            raise SystemExit(f'{task}: timeout. Cached files kept. Read the periodic stack trace above '
                             'to distinguish import, network and cache-lock waits. Repeat to retry.')
    paths=[root/f'{task}.json' for task in cfg['tasks']]
    if all(path.exists() for path in paths):
        reports=[json.loads(path.read_text()) for path in paths]
        hashes={r['task_source_sha256'] for r in reports}
        if len(hashes)!=1:
            raise ValueError('Task definitions changed between downloads; do not combine manifests.')
        merged={'datasets':{},'task_source_sha256':hashes.pop()}
        for report in reports:
            merged['datasets'].update(report['datasets'])
        write(root/'benchmark_data.json',merged)
        # Only a complete offline verification unlocks the full training pipeline.
        if args.action=='check' and set(tasks)==set(cfg['tasks']):
            write(root/'offline_ready.json',{'complete':True,'tasks':cfg['tasks'],
                  'manifest_sha256':hashlib.sha256((root/'benchmark_data.json').read_bytes()).hexdigest(),
                  'checked_at':time.strftime('%Y-%m-%d %H:%M:%S'),
                  'note':'Manifest only; actual datasets remain in the existing Hugging Face cache.'})
        print(f'{args.action} finished: {root}',flush=True)
    else:
        print('Requested subset finished; prepare the remaining tasks before full training.',flush=True)


if __name__=='__main__':
    main()
