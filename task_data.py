"""Visible, task-wise benchmark loading and offline fingerprint validation."""
import gc
import hashlib
import json
from pathlib import Path
import time


def snapshots(tasks):
    print('[data] importing lm_eval (no GPU model loading)',flush=True)
    from lm_eval.tasks import TaskManager, get_task_dict
    import lm_eval
    print('[data] indexing benchmark definitions',flush=True)
    manager=TaskManager()
    result={}
    def visit(node):
        for name,task in node.items():
            if isinstance(task,dict):
                visit(task)
            elif hasattr(task,'dataset'):
                result[str(name)]={str(split):{'rows':len(data),'fingerprint':data._fingerprint}
                                   for split,data in task.dataset.items()}
            else:
                raise TypeError(f'Unsupported task node: {type(task)}')
    for task in tasks:
        start=time.perf_counter()
        print(f'[data] BEGIN {task}: loading dataset/cache',flush=True)
        tree=get_task_dict([task],task_manager=manager)
        visit(tree)
        del tree
        gc.collect()
        print(f'[data] DONE {task}: {time.perf_counter()-start:.1f}s',flush=True)
    package=Path(lm_eval.__file__).parent/'tasks'
    code_hash=hashlib.sha256()
    for p in sorted(package.rglob('*')):
        if p.is_file() and p.suffix in ('.yaml','.yml','.py'):
            code_hash.update(str(p.relative_to(package)).encode())
            code_hash.update(p.read_bytes())
    return {'datasets':result,'task_source_sha256':code_hash.hexdigest()}


def prepare(tasks,root):
    # Training runner is offline: the separate data-preparation command creates this.
    path=root/'benchmark_data.json'
    if not path.exists():
        raise FileNotFoundError('Benchmark preparation is required: run prepare_eval_data.py download, then check.')
    verify(tasks,root)


def verify(tasks,root):
    path=root/'benchmark_data.json'
    if not path.exists() or json.loads(path.read_text()) != snapshots(tasks):
        raise ValueError('Benchmark dataset/source fingerprints differ from offline preparation.')
    print('[data] offline fingerprints verified',flush=True)
