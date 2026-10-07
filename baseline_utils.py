#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate native OLMoE / Qwen1.5-MoE-A2.7B checkpoints on prepared DCLM.

Single GPU, BF16, no training. Uses the checkpoint's native routing and top-k.
Validates tokenizer/data provenance. Qwen's shared expert is counted separately
from routed experts. Saves only small metrics, not checkpoint weights.

  python -u baseline_utils.py --model olmoe --config protocol.json
  python -u baseline_utils.py --model qwen --config protocol.json

Default --split valid. Reserve --split test for a fixed final protocol.
Default --benchmark-steps 0: no latency claim from concurrent evaluations.
Optional --benchmark-steps 30 measures repeated full-sequence forward passes
on one resident GPU batch, WITHOUT hooks, CE, H2D, KV cache or decoding.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from project_config import load_config, DEFAULT_CONFIG, PROJECT_ROOT

MODELS = {
    "olmoe": (str(PROJECT_ROOT / "models/olmoe"), "olmoe"),
    "qwen": (str(PROJECT_ROOT / "models/qwen"), "qwen2_moe"),
}
DATA_ROOT = PROJECT_ROOT / "data/processed"
RESULT_ROOT = PROJECT_ROOT / "outputs/baseline"
GIB = 1024 ** 3


def log(text):
    print(text, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_data(args):
    path = args.data_dir / f"{args.split}.npy"
    provenance = load_json(args.data_dir / "report.json")
    if provenance["recipe"]["model_name"] != args.model:
        raise ValueError("数据所属模型不匹配：不能混用 OLMoE 和 Qwen 的 token ID。")
    expected = provenance["splits"][args.split]["file_hashes"][path.name]
    actual = sha256(path)
    if expected != actual:
        raise ValueError("数据文件校验失败，请使用 prepare_data.py check 检查。")
    for name, digest in provenance["recipe"]["tokenizer_files"].items():
        # Model architecture can be recorded separately; tokenizer must match.
        if name == "config.json":
            continue
        local = args.model_path / name
        if not local.is_file() or sha256(local) != digest:
            raise ValueError(f"tokenizer 文件不匹配: {local}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.dtype.kind not in "iu" or not array.size:
        raise ValueError("数据必须是非空的二维整数数组。")
    if list(array.shape) != provenance["splits"][args.split]["shape"]:
        raise ValueError("数据 shape 与准备记录不一致。")
    count = min(args.max_sequences or len(array), len(array))
    if array.shape[1] < 2:
        raise ValueError("序列至少需要两个 token。")
    return array, count, {
        "path": str(path), "sha256": actual,
        "preparation_report_sha256": sha256(args.data_dir / "report.json"),
        "preparation_recipe": provenance["recipe"],
        "sequences_available": len(array), "sequences_evaluated": count,
        "selection": "first n packed blocks; no shuffle",
    }


def checkpoint_identity(directory):
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        weights = sorted(directory.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"未找到模型权重: {directory}")
    return {
        "path": str(directory),
        "config_sha256": sha256(directory / "config.json"),
        "index_sha256": {p.name: sha256(p) for p in directory.glob("*.index.json")},
        "weight_files": [
            {"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in weights
        ],
        "weight_identity_note": "Weight file size/mtime only, not full weight checksums.",
    }


def iter_moe_blocks(model):
    import torch
    found = []
    for index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            continue
        if not isinstance(mlp.experts, torch.nn.ModuleList):
            raise TypeError(
                "当前路由统计适用于 ModuleList 专家实现；此安装版本使用不同实现。"
                "请提供 transformers 版本，不要替换模型 forward 来绕过检查。"
            )
        n = len(mlp.experts)
        k = int(getattr(mlp, "top_k", model.config.num_experts_per_tok))
        if not 1 <= k <= n:
            raise ValueError(f"第 {index} 层 top-k 配置非法")
        found.append((index, mlp, n, k))
    if not found:
        raise ValueError("未识别到标准 Hugging Face MoE 层。")
    return found


class RoutingCounter:
    """Count actual expert invocations; never recompute top-k selections."""
    def __init__(self, blocks):
        self.blocks = blocks
        self.counts = [np.zeros(n, dtype=np.int64) for _, _, n, _ in blocks]
        self.shared_counts = np.zeros(len(blocks), dtype=np.int64)
        self.handles = []

    def attach(self):
        for row, (_, moe, _, _) in enumerate(self.blocks):
            for expert_id, expert in enumerate(moe.experts):
                def record(module, inputs, r=row, e=expert_id):
                    x = inputs[0]
                    self.counts[r][e] += x.numel() // x.shape[-1]
                self.handles.append(expert.register_forward_pre_hook(record))
            shared = getattr(moe, "shared_expert", None)
            if shared is not None:
                def record_shared(module, inputs, r=row):
                    x = inputs[0]
                    self.shared_counts[r] += x.numel() // x.shape[-1]
                self.handles.append(shared.register_forward_pre_hook(record_shared))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def summarize(self, input_tokens):
        rows = []
        for row, ((layer, moe, n, k), counts) in enumerate(zip(self.blocks, self.counts)):
            expected = input_tokens * k
            if int(counts.sum()) != expected:
                raise RuntimeError(
                    f"第 {layer} 层实际路由计数={int(counts.sum())}，预期={expected}。"
                    "专家实现可能绕过了统计钩子，停止发布指标。"
                )
            shared = getattr(moe, "shared_expert", None) is not None
            expected_shared = input_tokens if shared else 0
            if int(self.shared_counts[row]) != expected_shared:
                raise RuntimeError(f"第 {layer} 层共享专家计数不符。")
            probabilities = counts.astype(np.float64) / expected
            positive = probabilities > 0
            entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
            rows.append({
                "layer": layer, "num_routed_experts": n, "top_k": k,
                "load_cv": float(counts.std(ddof=0) / counts.mean()),
                "routing_entropy": entropy, "effective_experts": math.exp(entropy),
                "experts_observed": int(np.count_nonzero(counts)),
                "assignment_count": int(counts.sum()),
                "has_shared_expert": shared,
                "shared_expert_input_tokens": int(self.shared_counts[row]),
                "norm_topk_prob": bool(getattr(moe, "norm_topk_prob", False)),
            })
        return {
            "definition": "CV/entropy use routed-expert assignment counts only. "
                          "Shared experts are excluded and checked separately.",
            "mean_load_cv": float(np.mean([x["load_cv"] for x in rows])),
            "mean_effective_experts": float(np.mean([x["effective_experts"] for x in rows])),
            "layers": rows,
        }


def block_nll_sums(logits, ids, chunk_size):
    import torch
    import torch.nn.functional as F
    if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(ids.shape):
        raise RuntimeError("模型必须返回所有输入位置的 logits。")
    batch, length = ids.shape
    result = torch.zeros(batch, dtype=torch.float64, device=ids.device)
    for start in range(0, length - 1, chunk_size):
        end = min(start + chunk_size, length - 1)
        # Exactly one standard causal shift; model loss/auxiliary loss are unused.
        scores = logits[:, start:end].reshape(-1, logits.shape[-1]).float()
        targets = ids[:, start + 1:end + 1].reshape(-1)
        loss = F.cross_entropy(scores, targets, reduction="none")
        result += loss.reshape(batch, -1).sum(dim=1, dtype=torch.float64)
    return result


def forward(model, ids):
    return model(
        input_ids=ids, use_cache=False, output_router_logits=False,
        output_hidden_states=False, output_attentions=False, return_dict=True,
    )


def make_batch(data, start, stop, device):
    import torch
    # Small token batches: .tolist avoids the torch 2.3 / NumPy 2 bridge issue.
    return torch.tensor(data[start:stop].tolist(), dtype=torch.long, device=device)


def benchmark(model, data, count, args, device):
    import torch
    if args.benchmark_steps == 0:
        return None
    ids = make_batch(data, 0, min(count, args.batch_size), device)
    with torch.inference_mode():
        for _ in range(args.warmup_steps):
            output = forward(model, ids)
            del output
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.benchmark_steps):
            output = forward(model, ids)
            del output
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    result = {
        "kind": "repeated fixed-batch full-sequence forward with vocabulary projection",
        "includes": "model forward only; tokens already on GPU",
        "excludes": "hooks, cross entropy, H2D, KV cache, autoregressive decoding",
        "warmup_steps": args.warmup_steps, "steps": args.benchmark_steps,
        "batch_size": len(ids), "sequence_length": ids.shape[1],
        "seconds": elapsed,
        "input_tokens_per_second": ids.numel() * args.benchmark_steps / elapsed,
        "milliseconds_per_batch": elapsed * 1000 / args.benchmark_steps,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "caution": "Not decoding throughput. Other jobs can affect measurements; "
                   "use an idle GPU/node for reported efficiency comparisons.",
    }
    del ids
    return result


def save_routing(output, counter, routing, input_tokens):
    metrics = routing["layers"]
    with (output / "layer_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output / "expert_usage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "expert", "routed_tokens", "assignment_share",
                         "token_activation_rate"])
        for (layer, _, _, k), counts in zip(counter.blocks, counter.counts):
            for expert, value in enumerate(counts):
                writer.writerow([layer, expert, int(value),
                                 value / (input_tokens * k), value / input_tokens])


def run(args, output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("此脚本用于单张 CUDA GPU。")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("需要支持 BF16 的 GPU。")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data, count, data_info = validate_data(args)
    identity = checkpoint_identity(args.model_path)
    source_config = load_json(args.model_path / "config.json")
    if source_config.get("model_type") != MODELS[args.model][1]:
        raise ValueError(f"模型类型与 --model 不符: {source_config.get('model_type')}")
    length = data.shape[1]
    if length > source_config["max_position_embeddings"]:
        raise ValueError("数据长度超过检查点上下文长度。")
    token_min = int(data[:count].min())
    token_max = int(data[:count].max())
    if token_min < 0 or token_max >= source_config["vocab_size"]:
        raise ValueError("数据 token ID 超出模型词表。")
    log(f"模型: {args.model}; split={args.split}; {count} 条 × {length} tokens")
    log(f"GPU: {torch.cuda.get_device_name(device)}")
    log(f"当前空闲显存: {torch.cuda.mem_get_info(device)[0] / GIB:.2f} GiB")
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16, attn_implementation=args.attention,
        low_cpu_mem_usage=False,
    )
    model.to(device).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    blocks = iter_moe_blocks(model)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - start
    configurations = sorted({(n, k) for _, _, n, k in blocks})
    shared_layers = sum(getattr(m, "shared_expert", None) is not None for _, m, _, _ in blocks)
    log(f"MoE 层数={len(blocks)}; (专家数, top-k)={configurations}; "
        f"带共享专家层数={shared_layers}")
    timing = benchmark(model, data, count, args, device)
    counter = RoutingCounter(blocks)
    sums = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    counter.attach()
    try:
        with torch.inference_mode():
            for offset in range(0, count, args.batch_size):
                end = min(offset + args.batch_size, count)
                ids = make_batch(data, offset, end, device)
                result = forward(model, ids)
                losses = block_nll_sums(result.logits, ids, args.ce_chunk)
                values = losses.cpu().tolist()
                if not np.isfinite(values).all():
                    raise RuntimeError("NLL 出现 NaN/Inf，停止发布结果。")
                sums.extend(values)
                del losses, result, ids
                if end // args.log_every != offset // args.log_every or end == count:
                    elapsed = time.perf_counter() - start
                    mean = float(np.sum(sums, dtype=np.float64) / (end * (length - 1)))
                    eta = elapsed * (count - end) / end
                    log(f"已评估 {end}/{count}; NLL={mean:.6f}; "
                        f"PPL={math.exp(mean):.6f}; 预计剩余={eta / 60:.1f} 分钟")
                    save_json(output / "progress.json", {
                        "complete": False, "sequences": end, "total_sequences": count,
                        "mean_nll_so_far": mean, "elapsed_seconds": elapsed,
                    })
    finally:
        counter.remove()
    torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) / GIB
    predicted = count * (length - 1)
    input_tokens = count * length
    routing = counter.summarize(input_tokens)
    sums = np.asarray(sums, dtype=np.float64)
    mean = float(sums.sum(dtype=np.float64) / predicted)
    report = {
        "complete": True, "role": "supplied checkpoint evaluated without expansion/training",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint": identity, "model_config": model.config.to_dict(), "data": data_info,
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda, "python_version": sys.version,
        "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parameter_dtype": "bfloat16", "total_parameters": sum(p.numel() for p in model.parameters()),
        "moe_configurations": configurations, "shared_expert_layers": shared_layers,
        "sequence_length": length, "sequences": count, "input_tokens": input_tokens,
        "prediction_tokens": predicted, "mean_nll": mean, "ppl": math.exp(mean),
        "ppl_protocol": "Natural-log CE in FP32, accumulation in FP64. Standard causal shift "
                        "within disjoint packed blocks; first token not scored; no padding, "
                        "boundary masks, chat templates or auxiliary loss. EOS is scored.",
        "block_nll_definition": "block_nll.npy: mean NLL per predicted token in each block; "
                                "block_nll_sum.npy: summed NLL per block",
        "routing": routing, "load_seconds": load_seconds,
        "evaluation_seconds": evaluation_seconds,
        "evaluation_peak_allocated_gib": peak,
        "evaluation_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "evaluation_memory_scope": "Model, activations, full logits, CE chunks, routing hooks.",
        "forward_benchmark": timing, "script_sha256": sha256(Path(__file__)),
        "implementation": {},
    }
    for cls in {type(model), *(type(moe) for _, moe, _, _ in blocks)}:
        source = inspect.getsourcefile(cls)
        report["implementation"][cls.__module__ + "." + cls.__name__] = {
            "source_sha256": sha256(source) if source and Path(source).is_file() else None,
        }
    np.save(output / "block_nll_sum.npy", sums, allow_pickle=False)
    np.save(output / "block_nll.npy", sums / (length - 1), allow_pickle=False)
    save_routing(output, counter, routing, input_tokens)
    save_json(output / "report.json", report)
    save_json(output / "progress.json", {
        "complete": True, "sequences": count, "mean_nll": mean, "ppl": report["ppl"],
    })
    log(f"\n评估完成: {args.model}")
    log(f"NLL: {mean:.8f}\nPPL: {report['ppl']:.8f}")
    log(f"平均路由专家负载 CV: {routing['mean_load_cv']:.8f}")
    log(f"共享专家层数: {shared_layers}（不计入上述 CV）")
    log(f"评估耗时: {evaluation_seconds / 60:.2f} 分钟")
    log(f"评估峰值已分配显存: {peak:.2f} GiB")
    if timing:
        log(f"独立整段前向测速: {timing['input_tokens_per_second']:.2f} input tokens/s")
    log(f"结果目录: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--ce-chunk", type=int, default=256)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.ce_chunk, args.log_every, args.warmup_steps) < 1:
        parser.error("batch-size、ce-chunk、log-every、warmup-steps 必须为正。")
    if min(args.max_sequences, args.benchmark_steps, args.seed) < 0:
        parser.error("max-sequences、benchmark-steps、seed 必须非负。")
    cfg = load_config(args.config)
    args.model_path = (args.model_path or Path(cfg['models'][args.model]['path'])).expanduser().resolve()
    args.data_dir = (args.data_dir or Path(cfg['models'][args.model]['data'])).expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main():
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    size = f"n{args.max_sequences}" if args.max_sequences else "full"
    output = args.output_dir / f"{args.model}_{args.split}_{size}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args, output)
    except KeyboardInterrupt:
        save_json(output / "failure.json", {"complete": False, "reason": "interrupted"})
        log("已中断；此目录没有完整评估结果。")
        return 130
    except Exception as exc:
        save_json(output / "failure.json", {
            "complete": False, "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
