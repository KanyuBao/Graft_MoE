#!/usr/bin/env python3
"""Prepare document-disjoint DCLM token blocks for both base models (CPU only).

  python prepare_data.py prepare --models olmoe qwen
  python prepare_data.py prepare --local-jsonl /path/to/dclm.jsonl --models olmoe qwen
  python prepare_data.py check --models olmoe qwen

The default source is the official DCLM parquet release. Only a bounded stream is
read; this does not download the full corpus. The raw cache is reusable offline.
Repeat an interrupted command unchanged to resume. Changed recipes require new
raw/output directories. Raw-document hash splitting precedes tokenization; exact
text duplicates and repeated source IDs are removed. This is not near-deduplication
and does not establish decontamination against the base models' pretraining data.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

from preparation_common import (
    HERE, MODEL_REPOS, directory_lock, load_config, read_json, resolve_config_path,
    sha256, tokenizer_files, verify_files, write_json,
)

DATASET = "mlfoundations/dclm-baseline-1.0-parquet"
SPLITS = ("train", "valid", "test")
SCHEMA_VERSION = 1


def document_split(text_hash, seed, blocks):
    """One document always belongs to one split, before packing or truncation."""
    value = int(hashlib.sha256(f"{seed}:{text_hash}".encode()).hexdigest(), 16)
    bucket = value % sum(blocks.values())
    for split in SPLITS:
        if bucket < blocks[split]:
            return split
        bucket -= blocks[split]
    raise AssertionError("Unreachable split bucket")


def encode_document(tokenizer, text):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    # A single explicit EOS separates adjacent raw documents within a split.
    tokens.append(tokenizer.eos_token_id)
    return tokens


def local_rows(paths):
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_no}")
                yield row


def shuffled_rows(rows, size, seed):
    rng = random.Random(seed)
    iterator = iter(rows)
    buffer = list(itertools.islice(iterator, size))
    for row in iterator:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    rng.shuffle(buffer)
    yield from buffer


def source_rows(args, source):
    if source["kind"] == "local_jsonl":
        return shuffled_rows(local_rows(args.local_jsonl), args.shuffle_buffer, args.seed)
    if args.offline:
        raise RuntimeError("The raw cache is incomplete. Resume online or supply local JSONL in a new raw directory.")
    from datasets import load_dataset
    dataset = load_dataset(source["dataset"], name=source["config"],
                           split=source["split"], revision=source["resolved_revision"],
                           streaming=True)
    return iter(dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer))


def source_request(args):
    if args.local_jsonl:
        return {"kind": "local_jsonl", "files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in args.local_jsonl]}
    return {"kind": "huggingface", "dataset": args.dataset,
            "config": args.dataset_config, "split": args.dataset_split,
            "requested_revision": args.dataset_revision}


def verify_raw(root, manifest):
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete raw manifest: {root}")
    verify_files(root, manifest["files_sha256"])


def collect_raw(args, root, tokenizers, assets, blocks):
    """Crash-safe raw cache: commit offsets/hashes after fsync, then resume replay."""
    root.mkdir(parents=True, exist_ok=True)
    # A completed raw cache is self-contained: offline reuse need not retain the
    # original JSONL paths or re-enter a previously pinned dataset revision.
    cached_manifest = root / "raw_manifest.json"
    source = (read_json(cached_manifest)["request"]["source"]
              if args.offline and not args.local_jsonl and cached_manifest.exists()
              else source_request(args))
    request = {
        "schema_version": SCHEMA_VERSION, "source": source,
        "models": {key: {"tokenizer_files": assets[key], "eos_token_id": tok.eos_token_id}
                   for key, tok in tokenizers.items()},
        "block_size": args.block_size, "blocks": blocks, "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer, "text_field": args.text_field,
        "id_field": args.id_field,
        "splitting": "sha256(seed:sha256(UTF-8 text)) modulo sum(block quotas); train,valid,test",
        "deduplication": "exact UTF-8 text SHA-256 and nonempty source document ID; no near-deduplication",
    }
    with directory_lock(root / ".prepare.lock"):
        manifest_path = root / "raw_manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["request"] != request:
                raise ValueError(f"Raw-cache recipe differs: {root}. Use a new --raw-dir.")
            verify_raw(root, manifest)
            print(f"[raw ready] {root}", flush=True)
            return manifest
        state_path = root / "raw_in_progress.json"
        if state_path.exists():
            state = read_json(state_path)
            if state["request"] != request:
                raise ValueError(f"Interrupted raw-cache recipe differs: {root}. Use a new --raw-dir.")
        else:
            if any(p.name != ".prepare.lock" for p in root.iterdir()):
                raise ValueError(f"Unmanaged files in raw directory: {root}; use an empty directory.")
            source = dict(request["source"])
            if source["kind"] == "huggingface":
                if args.offline:
                    raise RuntimeError("--offline requires a completed raw cache or --local-jsonl.")
                from huggingface_hub import HfApi
                source["resolved_revision"] = HfApi().dataset_info(
                    args.dataset, revision=args.dataset_revision).sha
            state = {"request": request, "source": source, "next_source_row": 0,
                     "duplicates_skipped": 0, "empty_skipped": 0,
                     "counts": {s: {k: 0 for k in tokenizers} for s in SPLITS},
                     "documents": {s: 0 for s in SPLITS},
                     "offsets": {s: 0 for s in SPLITS},
                     "prefix_sha256": {s: hashlib.sha256(b"").hexdigest() for s in SPLITS}}
            write_json(state_path, state)
        seen_text, seen_ids = set(), set()
        hashers, streams = {}, {}
        try:
            for split in SPLITS:
                path = root / f"{split}.jsonl"
                if not path.exists() and state["offsets"][split]:
                    raise FileNotFoundError(f"Missing committed raw file: {path}")
                stream = path.open("r+b" if path.exists() else "w+b")
                streams[split] = stream
                if stream.seek(0, os.SEEK_END) < state["offsets"][split]:
                    raise ValueError(f"Truncated committed raw data: {path}")
                # Discard only uncommitted bytes from an interrupted append.
                stream.truncate(state["offsets"][split])
                stream.seek(0)
                hasher = hashlib.sha256()
                recovered_counts = {k: 0 for k in tokenizers}
                recovered_docs = 0
                for line in stream:
                    hasher.update(line)
                    row = json.loads(line)
                    if row["text_sha256"] in seen_text or (row["source_id"] and row["source_id"] in seen_ids):
                        raise ValueError(f"Duplicate document in raw cache: {path}")
                    seen_text.add(row["text_sha256"])
                    if row["source_id"]:
                        seen_ids.add(row["source_id"])
                    for key in tokenizers:
                        recovered_counts[key] += row["token_counts"][key]
                    recovered_docs += 1
                if hasher.hexdigest() != state["prefix_sha256"][split] or recovered_counts != state["counts"][split] or recovered_docs != state["documents"][split]:
                    raise ValueError(f"Raw-cache committed prefix is inconsistent: {path}")
                hashers[split] = hasher

            def commit():
                for split, stream in streams.items():
                    stream.flush()
                    os.fsync(stream.fileno())
                    state["offsets"][split] = stream.tell()
                    state["prefix_sha256"][split] = hashers[split].hexdigest()
                write_json(state_path, state)

            def full(split):
                target = blocks[split] * args.block_size
                return all(state["counts"][split][key] >= target for key in tokenizers)

            if not all(full(s) for s in SPLITS):
                print(f"[raw] resume source row {state['next_source_row']}; quotas {blocks}", flush=True)
                rows = itertools.islice(source_rows(args, state["source"]), state["next_source_row"], None)
                for row in rows:
                    if args.max_documents and state["next_source_row"] >= args.max_documents:
                        break
                    state["next_source_row"] += 1
                    text = row.get(args.text_field)
                    if not isinstance(text, str):
                        raise ValueError(f"Source row {state['next_source_row']} lacks string field {args.text_field!r}")
                    if not text.strip():
                        state["empty_skipped"] += 1
                    else:
                        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        source_id = str(row.get(args.id_field) or "")
                        split = document_split(digest, args.seed, blocks)
                        if digest in seen_text or (source_id and source_id in seen_ids):
                            state["duplicates_skipped"] += 1
                        elif not full(split):
                            counts = {key: len(encode_document(tok, text)) for key, tok in tokenizers.items()}
                            record = {"text": text, "text_sha256": digest, "source_id": source_id,
                                      "url": str(row.get("url") or ""), "source_row": state["next_source_row"] - 1,
                                      "split": split, "token_counts": counts}
                            line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                            streams[split].write(line)
                            hashers[split].update(line)
                            seen_text.add(digest)
                            if source_id:
                                seen_ids.add(source_id)
                            state["documents"][split] += 1
                            for key, count in counts.items():
                                state["counts"][split][key] += count
                    if state["next_source_row"] % args.commit_every == 0:
                        commit()
                        print(f"[raw] rows={state['next_source_row']} token_counts={state['counts']}", flush=True)
                    if all(full(s) for s in SPLITS):
                        break
            commit()
            if not all(full(s) for s in SPLITS):
                raise RuntimeError(f"Source exhausted or --max-documents reached before quotas: {state['counts']}. "
                                   "No training report was published. Resume with the same source/recipe and a larger cap.")
            manifest = {**state, "complete": True,
                        "files_sha256": {f"{s}.jsonl": hashers[s].hexdigest() for s in SPLITS}}
            write_json(manifest_path, manifest)
            state_path.unlink()
            return manifest
        finally:
            for stream in streams.values():
                stream.close()


def pack_split(raw_path, tokenizer, target, block_size, output, doc_manifest):
    """Append EOS per document, concatenate within split, then retain exact quota."""
    temporary = output.with_name(output.name + ".partial")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int32, shape=(target, block_size))
    flat = array.reshape(-1)
    filled = 0
    docs = 0
    manifest_temp = doc_manifest.with_name(doc_manifest.name + ".partial")
    with raw_path.open(encoding="utf-8") as stream, manifest_temp.open("w", encoding="utf-8") as records:
        for line in stream:
            row = json.loads(line)
            tokens = encode_document(tokenizer, row["text"])
            take = min(len(tokens), len(flat) - filled)
            if take:
                if min(tokens[:take]) < 0 or max(tokens[:take]) >= len(tokenizer):
                    raise ValueError("Tokenizer produced out-of-range token IDs")
                flat[filled:filled + take] = tokens[:take]
                records.write(json.dumps({"text_sha256": row["text_sha256"], "source_id": row["source_id"],
                                          "source_row": row["source_row"], "token_start": filled,
                                          "token_end": filled + take,
                                          "document_tokens_including_eos": len(tokens),
                                          "document_tail_truncated": take < len(tokens)}, ensure_ascii=False) + "\n")
                filled += take
                docs += 1
            if filled == len(flat):
                break
        records.flush()
        os.fsync(records.fileno())
    if filled != len(flat):
        raise ValueError(f"Insufficient raw tokens in {raw_path}: {filled} < {len(flat)}")
    array.flush()
    del flat, array
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    os.replace(manifest_temp, doc_manifest)
    return {"shape": [target, block_size], "dtype": "int32", "tokens": target * block_size,
            "documents_used": docs,
            "file_hashes": {output.name: sha256(output), doc_manifest.name: sha256(doc_manifest)}}


def check_prepared(root, key, model_path):
    report = read_json(root / "report.json")
    if not report.get("complete") or report["recipe"]["model_name"] != key:
        raise ValueError(f"Incomplete or wrong-model data report: {root}")
    verify_files(model_path, report["recipe"]["tokenizer_files"])
    verify_files(root / "tokenizer", report["recipe"]["tokenizer_files"])
    for split in (*SPLITS, "calibration"):
        entry = report["splits"][split]
        verify_files(root, entry["file_hashes"])
        array = np.load(root / f"{split}.npy", allow_pickle=False, mmap_mode="r")
        if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
            raise ValueError(f"Unexpected array layout: {root}/{split}.npy")
        if array.ndim != 2 or not array.size or int(array.min()) < 0 or int(array.max()) >= report["recipe"]["tokenizer_length"]:
            raise ValueError(f"Invalid token IDs: {root}/{split}.npy")
    verify_files(root, report["manifest_files_sha256"])
    train = np.load(root / "train.npy", mmap_mode="r", allow_pickle=False)
    calibration = np.load(root / "calibration.npy", mmap_mode="r", allow_pickle=False)
    indices = np.load(root / "calibration_train_indices.npy", allow_pickle=False)
    if len(indices) != len(calibration) or len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(train)).any():
        raise ValueError("Invalid calibration subset indices")
    if not np.array_equal(calibration, train[indices]):
        raise ValueError("Calibration is not the recorded training subset")
    # Verify exact text/ID disjointness in the documents that actually contributed tokens.
    seen_text, seen_ids = set(), set()
    for split in SPLITS:
        with (root / f"{split}_documents.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["text_sha256"] in seen_text or (row["source_id"] and row["source_id"] in seen_ids):
                    raise ValueError("Repeated text/source ID across packed documents")
                seen_text.add(row["text_sha256"])
                if row["source_id"]:
                    seen_ids.add(row["source_id"])
    print(f"[ready] {key}: {root}", flush=True)
    return report


def prepare_model(args, key, model_path, root, tokenizer, assets, raw_root, raw_manifest, blocks):
    recipe = {"schema_version": SCHEMA_VERSION, "model_name": key, "tokenizer_files": assets,
              "tokenizer_length": len(tokenizer), "eos_token_id": tokenizer.eos_token_id,
              "block_size": args.block_size, "blocks": {**blocks, "calibration": args.calibration_blocks},
              "seed": args.seed, "raw_manifest_sha256": sha256(raw_root / "raw_manifest.json"),
              "packing": "no automatic special tokens; one EOS after each document; no padding; exact token quota",
              "calibration": "seeded random training-block subset without replacement; shared with LM training",
              "implementation_sha256": sha256(Path(__file__))}
    with directory_lock(root.parent / f".{root.name}.prepare.lock"):
        if (root / "report.json").exists():
            report = check_prepared(root, key, model_path)
            if report["recipe"] != recipe:
                raise ValueError(f"Data recipe differs: {root}. Use a new data directory in protocol.json.")
            return
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Unmanaged or incomplete data directory: {root}; use a new empty destination.")
        staging = root.with_name(root.name + ".building")
        staging.mkdir(parents=True, exist_ok=True)
        request_path = staging / "recipe.json"
        if request_path.exists():
            if read_json(request_path) != recipe:
                raise ValueError(f"Interrupted packing recipe differs: {staging}. Use a new output directory.")
        elif any(staging.iterdir()):
            raise ValueError(f"Unmanaged staging directory: {staging}")
        else:
            write_json(request_path, recipe)
        tokenizer_root = staging / "tokenizer"
        tokenizer_root.mkdir(exist_ok=True)
        for name in assets:
            shutil.copyfile(model_path / name, tokenizer_root / name)
        reports = {}
        for split in SPLITS:
            completed = staging / f"{split}_packing.json"
            if completed.exists():
                reports[split] = read_json(completed)
                verify_files(staging, reports[split]["file_hashes"])
            else:
                print(f"[pack] {key}/{split}: {blocks[split]} x {args.block_size}", flush=True)
                reports[split] = pack_split(raw_root / f"{split}.jsonl", tokenizer, blocks[split], args.block_size,
                                           staging / f"{split}.npy", staging / f"{split}_documents.jsonl")
                write_json(completed, reports[split])
        indices = np.random.default_rng(args.seed).choice(blocks["train"], size=args.calibration_blocks, replace=False)
        indices = np.sort(indices).astype(np.int64)
        train = np.load(staging / "train.npy", mmap_mode="r", allow_pickle=False)
        np.save(staging / "calibration.npy", np.asarray(train[indices]), allow_pickle=False)
        np.save(staging / "calibration_train_indices.npy", indices, allow_pickle=False)
        reports["calibration"] = {"shape": [args.calibration_blocks, args.block_size], "dtype": "int32",
                                  "tokens": args.calibration_blocks * args.block_size,
                                  "file_hashes": {n: sha256(staging / n) for n in
                                                  ("calibration.npy", "calibration_train_indices.npy")}}
        shutil.copyfile(raw_root / "raw_manifest.json", staging / "raw_manifest.json")
        report = {"complete": True, "recipe": recipe, "splits": reports,
                  "source": raw_manifest["source"],
                  "manifest_files_sha256": {"raw_manifest.json": sha256(staging / "raw_manifest.json")},
                  "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(staging / "report.json", report)
        check_prepared(staging, key, model_path)
        if root.exists():
            root.rmdir()
        os.replace(staging, root)
        print(f"[complete] {key}: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("--config", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_REPOS), default=list(MODEL_REPOS))
    parser.add_argument("--raw-dir", type=Path, help="Default: data_preparation.raw_dir relative to config, or data/raw/dclm_subset")
    parser.add_argument("--local-jsonl", type=Path, nargs="+", help="Offline source documents (.jsonl or .jsonl.gz), each containing text")
    parser.add_argument("--offline", action="store_true", help="Never contact HF; requires complete raw cache or local JSONL")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-revision", default="main", help="Resolved to a pinned commit at first preparation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--train-blocks", type=int, default=16384)
    parser.add_argument("--valid-blocks", type=int, default=2048)
    parser.add_argument("--test-blocks", type=int, default=2048)
    parser.add_argument("--calibration-blocks", type=int, default=512)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--commit-every", type=int, default=250, help="Durably commit streaming progress every N source rows")
    parser.add_argument("--max-documents", type=int, default=0, help="Optional source-row cap (0: stop only when quotas are met)")
    args = parser.parse_args()
    positive = ("block_size", "train_blocks", "valid_blocks", "test_blocks", "calibration_blocks", "shuffle_buffer", "commit_every")
    if any(getattr(args, name) < 1 for name in positive) or args.block_size < 2:
        parser.error("Block counts/buffer/commit interval must be positive and block size at least two")
    if args.calibration_blocks > args.train_blocks or args.max_documents < 0 or args.seed < 0:
        parser.error("Calibration must fit in train; --max-documents and --seed must be nonnegative")
    config_path, cfg = load_config(args.config)
    keys = list(dict.fromkeys(args.models))
    model_paths = {k: resolve_config_path(config_path, cfg["models"][k]["path"]) for k in keys}
    data_paths = {k: resolve_config_path(config_path, cfg["models"][k]["data"]) for k in keys}
    if len(set(data_paths.values())) != len(keys):
        parser.error("Each model must have a distinct data output directory")
    if args.action == "check":
        for key in keys:
            check_prepared(data_paths[key], key, model_paths[key])
        return
    raw_root = args.raw_dir.expanduser().resolve() if args.raw_dir else resolve_config_path(
        config_path, cfg.get("data_preparation", {}).get("raw_dir", "data/raw/dclm_subset"))
    if args.local_jsonl:
        args.local_jsonl = [p.expanduser().resolve() for p in args.local_jsonl]
        if len(set(args.local_jsonl)) != len(args.local_jsonl) or any(not p.is_file() for p in args.local_jsonl):
            parser.error("--local-jsonl paths must be distinct existing files")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    tokenizers, assets = {}, {}
    for key in keys:
        assets[key] = tokenizer_files(model_paths[key])
        tokenizers[key] = AutoTokenizer.from_pretrained(str(model_paths[key]), local_files_only=True, trust_remote_code=False)
        if tokenizers[key].eos_token_id is None:
            raise ValueError(f"Tokenizer has no EOS token: {key}")
        # Raw documents may exceed the model context; packing, not tokenizer truncation, handles length.
        tokenizers[key].model_max_length = 10**30
    blocks = {s: getattr(args, f"{s}_blocks") for s in SPLITS}
    raw = collect_raw(args, raw_root, tokenizers, assets, blocks)
    for key in keys:
        prepare_model(args, key, model_paths[key], data_paths[key], tokenizers[key], assets[key], raw_root, raw, blocks)


if __name__ == "__main__":
    main()
