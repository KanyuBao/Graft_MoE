"""Shared configuration and portable project-relative paths."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / 'protocol.json'


def resolve_path(value, base):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return str((path if path.is_absolute() else Path(base) / path).resolve())


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as handle:
        cfg = json.load(handle)
    for field in ('output_root', 'benchmark_ready_root'):
        if field in cfg:
            cfg[field] = resolve_path(cfg[field], path.parent)
    for model in cfg['models'].values():
        for field in ('path', 'data', 'descriptors'):
            model[field] = resolve_path(model[field], path.parent)
    return cfg
