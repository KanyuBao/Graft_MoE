"""Small, CPU-only utilities shared by the download/preparation entry points."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_REPOS = {
    "olmoe": "allenai/OLMoE-1B-7B-0125",
    "qwen": "Qwen/Qwen1.5-MoE-A2.7B",
}
TOKENIZER_NAMES = (
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "tokenizer.model",
    "vocab.json", "vocab.txt", "merges.txt", "spiece.model",
    "chat_template.jinja",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def resolve_config_path(config_path, value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = read_json(path)
    for key in MODEL_REPOS:
        if key not in cfg.get("models", {}):
            raise ValueError(f"Missing models.{key} in {path}")
    return path, cfg


def tokenizer_files(path):
    result = {name: sha256(Path(path) / name) for name in TOKENIZER_NAMES
              if (Path(path) / name).is_file()}
    if "tokenizer_config.json" not in result or not any(
            name in result for name in ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt", "spiece.model")):
        raise FileNotFoundError(f"Incomplete local tokenizer: {path}. Download the model first.")
    return result


@contextmanager
def directory_lock(path):
    """An OS lock is released on crashes; the empty lock file may persist."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another preparation process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_files(root, expected):
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid manifest filename: {name}")
        path = Path(root) / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed file: {path}")
