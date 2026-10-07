#!/usr/bin/env python3
"""Download pinned base checkpoints, or verify their local manifests (no GPU).

  python download_models.py download --models olmoe qwen
  python download_models.py check --models olmoe qwen

Interrupted downloads resume on the same command. A completed directory is never
silently changed to a different repository or revision. HF_TOKEN and HF_HOME use
the normal Hugging Face conventions. This downloads base, not chat, checkpoints.
"""
import argparse
from pathlib import Path
import time

from preparation_common import (
    HERE, MODEL_REPOS, TOKENIZER_NAMES, directory_lock, load_config, read_json,
    resolve_config_path, sha256, tokenizer_files, verify_files, write_json,
)


def validate_structure(root, key):
    config = read_json(root / "config.json")
    expected = {"olmoe": ("olmoe", 64), "qwen": ("qwen2_moe", 60)}[key]
    if config.get("model_type") != expected[0] or config.get("num_experts") != expected[1]:
        raise ValueError(f"{root}: expected {expected[0]} with {expected[1]} routed experts")
    tokenizer_files(root)
    index = root / "model.safetensors.index.json"
    if index.exists():
        shards = set(read_json(index)["weight_map"].values())
        if not shards:
            raise ValueError(f"Empty weight index: {index}")
        for name in shards:
            if Path(name).name != name or not (root / name).is_file() or not (root / name).stat().st_size:
                raise ValueError(f"Missing model shard: {name}")
    elif not (root / "model.safetensors").is_file():
        raise FileNotFoundError(f"No complete safetensors checkpoint in {root}")


def check_model(root, key, repo):
    path = root / "download_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No completed download manifest: {path}")
    manifest = read_json(path)
    if not manifest.get("complete") or manifest.get("model") != key or manifest.get("repo_id") != repo:
        raise ValueError(f"Download manifest does not match requested model: {path}")
    verify_files(root, manifest["files_sha256"])
    validate_structure(root, key)
    print(f"[ready] {key}: {root} (revision {manifest['revision']})", flush=True)
    return manifest


def download_model(root, key, repo, requested_revision, workers):
    from huggingface_hub import HfApi, snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    with directory_lock(root / ".download.lock"):
        completed = root / "download_manifest.json"
        if completed.exists():
            manifest = check_model(root, key, repo)
            if requested_revision not in (None, manifest["revision"], manifest.get("requested_revision")):
                raise ValueError("Revision differs; choose a new model directory in protocol.json.")
            return
        building = root / "download_in_progress.json"
        if building.exists():
            plan = read_json(building)
            if plan["repo_id"] != repo or plan["model"] != key or (
                    requested_revision is not None and requested_revision not in
                    (plan["revision"], plan["requested_revision"])):
                raise ValueError("Interrupted download has a different recipe; use a new directory.")
        else:
            unmanaged = [p for p in root.iterdir() if p.name not in (".download.lock",)]
            if unmanaged:
                raise ValueError(f"Refusing to mix an unmanaged directory with a new download: {root}. "
                                 "Use an empty destination, or use existing checkpoints directly in protocol.json.")
            revision = requested_revision or "main"
            info = HfApi().model_info(repo, revision=revision)
            files = sorted(s.rfilename for s in info.siblings if
                           s.rfilename in set(TOKENIZER_NAMES) | {
                               "generation_config.json", "model.safetensors.index.json", "README.md",
                               "LICENSE", "LICENSE.txt"}
                           or ("/" not in s.rfilename and s.rfilename.endswith(".safetensors")))
            if not any(name.endswith(".safetensors") for name in files):
                raise ValueError(f"Repository contains no safetensors weights: {repo}")
            plan = {"model": key, "repo_id": repo, "requested_revision": revision,
                    "revision": info.sha, "files": files}
            write_json(building, plan)
        print(f"[download] {repo}@{plan['revision']} -> {root}", flush=True)
        snapshot_download(repo_id=repo, revision=plan["revision"], local_dir=str(root),
                          allow_patterns=plan["files"], max_workers=workers)
        for name in plan["files"]:
            if not (root / name).is_file():
                raise FileNotFoundError(f"Download is incomplete: {root / name}")
        validate_structure(root, key)
        print(f"[hash] {key}: recording SHA-256 for all downloaded files", flush=True)
        manifest = {**plan, "complete": True,
                    "files_sha256": {name: sha256(root / name) for name in plan["files"]},
                    "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(completed, manifest)
        building.unlink()
        print(f"[complete] {key}: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("download", "check"))
    parser.add_argument("--config", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_REPOS), default=list(MODEL_REPOS))
    parser.add_argument("--revision", help="Optional branch/tag/commit, applied to each requested model.")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    config_path, cfg = load_config(args.config)
    for key in dict.fromkeys(args.models):
        entry = cfg["models"][key]
        root = resolve_config_path(config_path, entry["path"])
        repo = entry.get("repo_id", MODEL_REPOS[key])
        if args.action == "check":
            check_model(root, key, repo)
        else:
            download_model(root, key, repo, args.revision or entry.get("revision"), args.max_workers)


if __name__ == "__main__":
    main()
