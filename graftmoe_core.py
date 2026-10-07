#!/usr/bin/env python3
"""OLMoE / Qwen2-MoE expansion, resumable training and native evaluation.
Optional training-only confidence routing; CPU FP32 master Adafactor.
"""
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import baseline_utils as bu

GIB = 1024 ** 3
METHODS = ('continue', 'random_copy', 'traffic_copy', 'eu_gn', 'graftmoe', 'graftmoe_native',
           'cluster_no_warm', 'random_groups_warm')
WARM_METHODS = ('graftmoe', 'graftmoe_native', 'random_groups_warm')
CIRA_METHODS = ('graftmoe', 'cluster_no_warm', 'random_groups_warm')
import confidence_routing as cira
TASKS = {'hellaswag': (0, 'acc_norm,none'), 'arc_challenge': (0, 'acc_norm,none'),
         'piqa': (0, 'acc,none'), 'winogrande': (0, 'acc,none'),
         'mmlu': (5, 'acc,none'), 'gsm8k': (8, 'exact_match,flexible-extract')}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tensor_array(t):
    return np.asarray(t.detach().float().cpu().tolist())


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.building')
    with temporary.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_array(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.building')
    with temporary.open('wb') as f:
        np.save(f, value, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def load_pt(path):
    # Only this package's own checkpoints or user-generated descriptors.
    return torch.load(path, map_location='cpu', weights_only=True)


def data_for(cfg, key, split):
    m = cfg['models'][key]
    args = SimpleNamespace(data_dir=Path(m['data']), split=split, model=key,
                           model_path=Path(m['path']), max_sequences=0)
    return bu.validate_data(args)[0]


def load_model(cfg, key, device):
    from transformers import AutoConfig, AutoModelForCausalLM
    path = cfg['models'][key]['path']
    c = AutoConfig.from_pretrained(path, local_files_only=True)
    expected = 'olmoe' if key == 'olmoe' else 'qwen2_moe'
    if c.model_type != expected:
        raise ValueError(f'{path}: model_type={c.model_type}, expected {expected}')
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32,
        attn_implementation=cfg['attention'], low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.to(device).eval().requires_grad_(False)
    for _, moe, n, k in bu.iter_moe_blocks(model):
        if n != cfg['models'][key]['old_experts'] or k != cfg['models'][key]['top_k']:
            raise ValueError('Native expert count/top-k differs from frozen protocol.')
    return model


def apply_expansion(model, plan, bias_noise=0., seed=42):
    blocks = bu.iter_moe_blocks(model)
    if len(blocks) != len(plan['layers']):
        raise ValueError('Expansion plan layer count mismatch.')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        for (idx, moe, n, k), row in zip(blocks, plan['layers']):
            if (idx, n, k) != (row['layer'], row['num_old_experts'], row['top_k']):
                raise ValueError('Expansion plan architecture mismatch.')
            parents = row['parents']
            if len(parents) != plan['expand_to'] - n or any(p < 0 or p >= n for p in parents):
                raise ValueError('Invalid expansion parents.')
            old_gate = moe.gate
            has_bias = old_gate.bias is not None or bias_noise > 0
            gate = torch.nn.Linear(old_gate.in_features, n + len(parents), bias=has_bias,
                                   device=old_gate.weight.device, dtype=old_gate.weight.dtype)
            gate.weight[:n].copy_(old_gate.weight)
            if gate.bias is not None:
                gate.bias.zero_()
                if old_gate.bias is not None:
                    gate.bias[:n].copy_(old_gate.bias)
            for j, p in enumerate(parents, n):
                moe.experts.append(copy.deepcopy(moe.experts[p]))
                gate.weight[j].copy_(old_gate.weight[p])
                if gate.bias is not None:
                    gate.bias[j].copy_(old_gate.bias[p] if old_gate.bias is not None else torch.zeros_like(gate.bias[j]))
                    if bias_noise:
                        noise = (2 * torch.rand((), generator=generator).item() - 1) * bias_noise
                        gate.bias[j].add_(noise)
            moe.gate = gate
            moe.num_experts = len(moe.experts)
            if moe.top_k != k:
                raise AssertionError('Native top-k must remain unchanged.')
    model.config.num_experts = plan['expand_to']
    # HF CausalLM wrappers cache this separately for their returned auxiliary loss.
    # We use our own layer-averaged LB, but the native forward still computes theirs.
    if hasattr(model, 'num_experts'):
        model.num_experts = plan['expand_to']
    model.requires_grad_(False)


def parameter_groups(model, method, old, lr):
    model.requires_grad_(False)
    if method == 'continue':
        model.requires_grad_(True)
    elif method != 'original':
        for _, moe, _, _ in bu.iter_moe_blocks(model):
            moe.gate.requires_grad_(True)
            for expert in moe.experts[old:]:
                expert.requires_grad_(True)
    # One CPU group per decoder layer. Non-layer parameters get separate groups.
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split('.')
        group = '.'.join(parts[:3]) if name.startswith('model.layers.') else parts[0] + '.' + parts[1]
        groups.setdefault(group, []).append((name, p))
    return [(name, entries, lr) for name, entries in groups.items()]


class MasterAdafactor:
    """Stock HF Adafactor, CPU FP32 masters; GPU model/accumulated grads BF16.
    Factorized second moment, no first moment; no quantized optimizer state.
    Copy gradients a group at a time. Exact resume includes all FP32 masters.
    """
    def __init__(self, groups, weight_decay):
        from transformers.optimization import Adafactor
        self.groups = []
        for name, entries, lr in groups:
            masters = [torch.nn.Parameter(p.detach().to(device='cpu', dtype=torch.float32, copy=True)) for _, p in entries]
            opt = Adafactor(masters, lr=lr, eps=(1e-30, 1e-3), clip_threshold=1.,
                            decay_rate=-0.8, beta1=None, weight_decay=weight_decay,
                            scale_parameter=False, relative_step=False, warmup_init=False)
            self.groups.append((name, entries, masters, opt, lr))

    def step(self, lr_factor):
        norm_sq = 0.
        for _, entries, masters, opt, lr in self.groups:
            opt.param_groups[0]['lr'] = lr * lr_factor
            for (_, p), master in zip(entries, masters):
                if p.grad is not None:
                    master.grad = p.grad.detach().to(device='cpu', dtype=torch.float32)
                    if not torch.isfinite(master.grad).all():
                        raise FloatingPointError('Non-finite gradient; checkpoint not advanced.')
                    norm_sq += master.grad.double().square().sum().item()
            opt.step()
            with torch.no_grad():
                for (_, p), master in zip(entries, masters):
                    p.copy_(master)
                    p.grad = None
            opt.zero_grad(set_to_none=True)
        return math.sqrt(norm_sq)

    def estimated_bytes(self):
        return int(sum(m.numel() * 4 for _, _, ms, _, _ in self.groups for m in ms) * 1.015) + 16 * 1024 ** 2

    def save(self, directory):
        files = {}
        for i, (name, entries, masters, opt, lr) in enumerate(self.groups):
            path = directory / f'group_{i:03d}.pt'
            atomic_torch(path, {'name': name, 'names': [n for n, _ in entries],
                                'masters': [m.detach() for m in masters], 'optimizer': opt.state_dict()})
            files[path.name] = {'sha256': bu.sha256(path), 'bytes': path.stat().st_size}
        return files

    def restore(self, directory, files):
        with torch.no_grad():
            for i, (name, entries, masters, opt, lr) in enumerate(self.groups):
                path = directory / f'group_{i:03d}.pt'
                if bu.sha256(path) != files[path.name]['sha256']:
                    raise ValueError(f'Checkpoint checksum failed: {path}')
                state = load_pt(path)
                if state['name'] != name or state['names'] != [n for n, _ in entries]:
                    raise ValueError('Checkpoint parameter order mismatch.')
                for (_, p), m, saved in zip(entries, masters, state['masters']):
                    m.copy_(saved)
                    p.copy_(m)
                opt.load_state_dict(state['optimizer'])
                del state


def retire_checkpoint(directory):
    """Free a validated old slot without exposing a half-deleted active name."""
    retired = directory.with_name(directory.name + '.retired')
    if retired.exists():
        shutil.rmtree(retired)
    if directory.exists():
        os.replace(directory, retired)
        shutil.rmtree(retired)


def save_checkpoint(out, optimizer, step, recipe_hash, train_seconds, device):
    required = optimizer.estimated_bytes() + 2 * GIB
    if shutil.disk_usage(out).free < required:
        raise OSError(f'Checkpoint needs {required/GIB:.1f} GiB FREE in {out}. '
                      'Existing committed checkpoint preserved; move output root or free space.')
    previous = bu.load_json(out / 'latest.json') if (out / 'latest.json').exists() else None
    slot = 1 - int(previous['slot']) if previous else 0
    building = out / f'checkpoint_{slot}.building'
    target = out / f'checkpoint_{slot}'
    # Only our uncommitted scratch namespace is removable here.
    if building.exists():
        shutil.rmtree(building)
    if target.exists():
        raise RuntimeError(f'Unexpected unreferenced checkpoint: {target}; inspect before deleting.')
    building.mkdir()
    files = optimizer.save(building)
    rng = {'torch_cpu': torch.get_rng_state(),
           'torch_cuda': torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None}
    atomic_torch(building / 'rng.pt', rng)
    files['rng.pt'] = {'sha256': bu.sha256(building / 'rng.pt')}
    meta = {'step': step, 'slot': slot, 'recipe_hash': recipe_hash, 'files': files,
            'train_seconds': train_seconds, 'complete': True}
    bu.save_json(building / 'manifest.json', meta)
    os.replace(building, target)
    bu.save_json(out / 'latest.json', meta)
    if previous:
        retire_checkpoint(out / f"checkpoint_{previous['slot']}")
    warm = out / 'warm_cache'
    if warm.exists():
        shutil.rmtree(warm)
    print(f'Committed checkpoint step={step}; previous slot removed.', flush=True)


def restore_checkpoint(out, optimizer, recipe_hash, device):
    latest = out / 'latest.json'
    if not latest.exists():
        return 0, 0.
    meta = bu.load_json(latest)
    if meta['recipe_hash'] != recipe_hash:
        raise ValueError('Resume recipe differs. Use a new output directory for changed experiments.')
    directory = out / f"checkpoint_{meta['slot']}"
    if bu.load_json(directory / 'manifest.json') != meta:
        raise ValueError('Committed checkpoint manifest mismatch.')
    optimizer.restore(directory, meta['files'])
    if bu.sha256(directory / 'rng.pt') != meta['files']['rng.pt']['sha256']:
        raise ValueError('RNG checkpoint checksum failed.')
    rng = load_pt(directory / 'rng.pt')
    torch.set_rng_state(rng['torch_cpu'])
    if rng['torch_cuda'] is not None:
        torch.cuda.set_rng_state(rng['torch_cuda'], device)
    return meta['step'], meta['train_seconds']


def recover_manifest(out, recipe_hash):
    """Recover a committed slot and retire older slots after an interrupted commit."""
    current = bu.load_json(out/'latest.json') if (out/'latest.json').exists() else None
    candidates = []
    obsolete = []
    for slot in (0, 1):
        retired = out/f'checkpoint_{slot}.retired'
        if retired.exists():
            shutil.rmtree(retired)
    for slot in (0, 1):
        path = out/f'checkpoint_{slot}'
        if not path.exists():
            continue
        if not (path/'manifest.json').exists():
            raise ValueError(f'Unrecognized checkpoint directory: {path}')
        meta = bu.load_json(path/'manifest.json')
        if (meta.get('recipe_hash') != recipe_hash or not meta.get('complete')
                or meta.get('slot') != slot):
            raise ValueError(f'Unrecognized checkpoint manifest: {path}')
        if current and slot == current['slot'] and meta != current:
            raise ValueError('Current checkpoint manifest differs from latest.json.')
        if current and meta['step'] < current['step']:
            obsolete.append(path)
            continue
        if all((path/name).is_file() and bu.sha256(path/name) == entry['sha256']
               for name, entry in meta['files'].items()):
            candidates.append(meta)
        else:
            raise ValueError(f'Incomplete or corrupt checkpoint: {path}')
    if current and not any(m == current for m in candidates):
        raise ValueError('The committed checkpoint is missing or invalid; files were preserved.')
    if candidates:
        chosen = max(candidates, key=lambda m: m['step'])
        if chosen != current:
            bu.save_json(out/'latest.json', chosen)
            print(f'Recovered complete snapshot at step {chosen["step"]}.', flush=True)
        obsolete += [out/f"checkpoint_{m['slot']}" for m in candidates if m['slot'] != chosen['slot']]
        for path in obsolete:
            retire_checkpoint(path)


def cleanup_completed_run(out):
    """Idempotently remove retired optimizer state only after a durable completion."""
    report = bu.load_json(out/'complete.json')
    if not report.get('complete'):
        return
    latest = out/'latest.json'
    if latest.exists():
        meta = bu.load_json(latest)
        if meta.get('recipe_hash') != report.get('recipe_hash'):
            raise ValueError('Completed result and optimizer checkpoint have different recipes.')
        bu.save_json(out/'retired_checkpoint.json', meta)
        path = out/f"checkpoint_{meta['slot']}"
        retire_checkpoint(path)
        latest.unlink()
    for slot in (0, 1):
        retired = out/f'checkpoint_{slot}.retired'
        if retired.exists():
            shutil.rmtree(retired)
    warm = out/'warm_cache'
    if warm.exists():
        shutil.rmtree(warm)


def lm_loss(logits, ids, chunk=128):
    loss = torch.zeros((), device=ids.device, dtype=torch.float32)
    count = ids.shape[0] * (ids.shape[1] - 1)
    for a in range(0, ids.shape[1] - 1, chunk):
        b = min(a + chunk, ids.shape[1] - 1)
        loss = loss + F.cross_entropy(logits[:, a:b].reshape(-1, logits.shape[-1]).float(),
                                      ids[:, a+1:b+1].reshape(-1), reduction='sum') / count
    return loss


def balance_loss(router_logits, blocks):
    if len(router_logits) != len(blocks):
        raise ValueError('Router logit layer count mismatch.')
    terms = []
    for logits, (_, moe, n, k) in zip(router_logits, blocks):
        probs = logits.reshape(-1, n).float().softmax(-1)
        chosen = getattr(moe, 'last_dispatch', None)
        if chosen is None:
            chosen = probs.detach().topk(k, dim=-1).indices
        freq = torch.bincount(chosen.reshape(-1), minlength=n).float() / chosen.numel()
        terms.append(n * (freq * probs.mean(0)).sum())
    # f uses fraction of assignments, not fraction of tokens. Uniform baseline=1.
    return torch.stack(terms).mean()


def lr_multiplier(step, total, warmup):
    if step <= warmup:
        return step / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def layer_output(moe, hidden):
    y = moe(hidden.unsqueeze(0))
    if not isinstance(y, tuple):
        raise TypeError('Expected native MoE (hidden, router_logits) output; inspect HF version.')
    return y[0].squeeze(0)


def warm_start(model, plan, descriptor_dir, out, cfg, seed, device):
    cache = out / 'warm_cache'
    cache.mkdir(exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for idx, moe, _, _ in bu.iter_moe_blocks(model):
        layer_start = time.perf_counter()
        saved_path = cache / f'layer_{idx:02d}.pt'
        old = plan['num_old_experts']
        params = list(moe.gate.parameters()) + [p for e in moe.experts[old:] for p in e.parameters()]
        if saved_path.exists():
            saved = load_pt(saved_path)
            with torch.no_grad():
                for p, value in zip(params, saved['values']):
                    p.copy_(value)
            rows.append(saved['metrics'])
            continue
        payload = load_pt(Path(descriptor_dir) / f'layer_{idx:02d}.pt')
        teacher = copy.deepcopy(moe).eval().requires_grad_(False)
        teacher.experts = torch.nn.ModuleList(list(teacher.experts[:old]))
        original_dtype = moe.gate.weight.dtype
        gate = torch.nn.Linear(moe.gate.in_features, old, bias=moe.gate.bias is not None,
                               device=device, dtype=original_dtype)
        with torch.no_grad():
            gate.weight.copy_(moe.gate.weight[:old])
            if gate.bias is not None:
                gate.bias.copy_(moe.gate.bias[:old])
        teacher.gate, teacher.num_experts = gate.requires_grad_(False), old
        fit = payload['fit']['hidden'].to(device=device, dtype=original_dtype)
        check = payload['check']['hidden'].to(device=device, dtype=original_dtype)
        del payload
        with torch.inference_mode():
            yfit = layer_output(teacher, fit).float()
            ycheck = layer_output(teacher, check).float()
            denom = yfit.square().mean().clamp_min(1e-12)
            check_denom = ycheck.square().mean().clamp_min(1e-12)
            before = ((layer_output(moe, check).float()-ycheck).square().mean()/check_denom).item()
        # Normal tensors are required when targets participate in autograd losses.
        yfit, ycheck = yfit.clone(), ycheck.clone()
        denom, check_denom = denom.clone(), check_denom.clone()
        for p in params:
            p.data = p.data.float()
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=cfg['warm_lr'], weight_decay=0., foreach=False)
        generator = torch.Generator().manual_seed(seed + idx * 7919)
        for _ in range(cfg['warm_steps']):
            ids = torch.randint(len(fit), (min(cfg['warm_batch'], len(fit)),), generator=generator).to(device)
            with (torch.autocast('cuda', dtype=torch.bfloat16)
                  if str(device).startswith('cuda') else nullcontext()):
                pred = layer_output(moe, fit[ids]).float()
                loss = (pred-yfit[ids]).square().mean() / denom
            if not torch.isfinite(loss):
                raise FloatingPointError('Warm-start loss is not finite.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step()
            opt.zero_grad(set_to_none=True)
        for p in params:
            p.data = p.data.to(original_dtype)
            p.requires_grad_(False)
        with torch.inference_mode():
            after = ((layer_output(moe, check).float()-ycheck).square().mean()/check_denom).item()
        row = {'layer': idx, 'check_nmse_before': before, 'check_nmse_after': after,
               'fit_hidden': len(fit), 'check_hidden': len(check), 'steps': cfg['warm_steps']}
        row['seconds'] = time.perf_counter() - layer_start
        atomic_torch(saved_path, {'values': [p.detach().cpu() for p in params], 'metrics': row})
        rows.append(row)
        print(f'Warm layer {idx:02d}: check NMSE {before:.6f} -> {after:.6f}', flush=True)
        del teacher, fit, check, yfit, ycheck, opt, params
        gc.collect()
        torch.cuda.empty_cache()
    model.requires_grad_(False)
    elapsed = time.perf_counter() - t0
    report = {'layers': rows, 'seconds_this_invocation': elapsed,
              'completed_layer_seconds': sum(row['seconds'] for row in rows),
              'note': 'CHECK diagnostics only; fixed warm steps, no check-based model selection.'}
    bu.save_json(out / 'warm_report.json', report)
    return report['completed_layer_seconds']


def evaluate(model, data, indices, out, label, old, device, step=0, train_seconds=0.):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if len(indices) == 0:
        raise ValueError('Evaluation requires at least one sequence.')
    path = out / (label + '.json')
    # Do not reuse by filename here: replayed uncommitted steps must be measured again.
    model.eval()
    blocks = bu.iter_moe_blocks(model)
    counter = bu.RoutingCounter(blocks)
    counter.attach()
    losses = []
    t0 = time.perf_counter()
    use_cuda = str(device).startswith('cuda')
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for j, ix in enumerate(indices):
                ids = bu.make_batch(data, int(ix), int(ix)+1, device)
                result = bu.forward(model, ids)
                value = bu.block_nll_sums(result.logits, ids, 128).item() / (ids.shape[1]-1)
                if not math.isfinite(value):
                    raise FloatingPointError('Evaluation NLL is non-finite.')
                losses.append(value)
                del result
                if (j+1) % 256 == 0:
                    print(f'{label}: {j+1}/{len(indices)} NLL={np.mean(losses):.6f}', flush=True)
    finally:
        counter.remove()
    if use_cuda:
        torch.cuda.synchronize(device)
    routing = counter.summarize(len(indices) * data.shape[1])
    shares = [float(c[old:].sum()/c.sum()) for c in counter.counts]
    record = {'label': label, 'step': step, 'train_seconds': train_seconds,
              'nll': float(np.mean(losses)), 'ppl': math.exp(float(np.mean(losses))),
              'blocks': len(indices), 'prediction_tokens': len(indices)*(data.shape[1]-1),
              'seconds': time.perf_counter()-t0, 'mean_load_cv': routing['mean_load_cv'],
              'new_assignment_share': float(np.mean(shares)),
              'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/GIB if use_cuda else None,
              'routing': routing, 'indices_sha256': digest([int(i) for i in indices])}
    atomic_array(out / (label + '_block_nll.npy'), np.asarray(losses, dtype=np.float64))
    atomic_array(out / (label + '_indices.npy'), np.asarray(indices, dtype=np.int64))
    atomic_array(out / (label + '_routing.npy'), np.stack(counter.counts))
    bu.save_json(path, record)
    print(f'{label}: NLL={record["nll"]:.6f} PPL={record["ppl"]:.6f} '
          f'new-share={record["new_assignment_share"]:.4f}', flush=True)
    return record


def run_tasks(model, key, cfg, out, device, benchmark_root=None, task_names=None):
    from task_data import verify
    verify(cfg['tasks'], Path(benchmark_root) if benchmark_root is not None else out.parent.parent)
    selected_tasks = cfg['tasks'] if task_names is None else task_names
    if any(task not in cfg['tasks'] for task in selected_tasks):
        raise ValueError('Requested task is not part of the frozen experiment protocol.')
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['models'][key]['path'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval().requires_grad_(False)
    model.config.use_cache = True
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, device=str(device),
              max_length=min(4096, int(getattr(model.config, 'max_position_embeddings', 4096))))
    reports = {}
    for task in selected_tasks:
        path = out / f'task_{task}.json'
        if path.exists():
            reports[task] = bu.load_json(path)
            continue
        shots, metric = TASKS[task]
        t0 = time.perf_counter()
        result = simple_evaluate(model=lm, tasks=[task], num_fewshot=shots,
                                 batch_size=1, limit=None, bootstrap_iters=1000,
                                 log_samples=True, apply_chat_template=False,
                                 random_seed=0, numpy_random_seed=1234,
                                 torch_random_seed=1234, fewshot_random_seed=1234)
        if metric not in result['results'][task]:
            raise ValueError(f'Expected predeclared metric {task}/{metric}; got {result["results"][task].keys()}')
        samples = result.pop('samples', {})
        # Harness results may contain numpy scalar / tensor values.
        def encode(value):
            if hasattr(value, 'tolist'):
                return value.tolist()
            if isinstance(value, set):
                return sorted(value)
            return str(value)
        raw = json.loads(json.dumps(result, default=encode))
        report = {'task': task, 'metric': metric, 'score': float(raw['results'][task][metric]),
                  'fewshot': shots, 'seconds': time.perf_counter()-t0, 'raw': raw}
        # Scores without per-example evidence are not marked complete.
        sample_path = out / f'task_{task}_samples.jsonl'
        tmp = sample_path.with_suffix('.jsonl.building')
        with tmp.open('w', encoding='utf-8') as f:
            for subtask, examples in samples.items():
                for example in examples:
                    f.write(json.dumps({'subtask': subtask, **example}, default=encode, ensure_ascii=False)+'\n')
        os.replace(tmp, sample_path)
        bu.save_json(path, report)
        reports[task] = report
    model.config.use_cache = False
    return reports


def inference_benchmark(model, data, device, steps):
    model.eval()
    ids = bu.make_batch(data, 0, 1, device)
    with torch.inference_mode():
        for _ in range(3):
            result = bu.forward(model, ids)
            del result
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(steps):
            result = bu.forward(model, ids)
            del result
        torch.cuda.synchronize(device)
        seconds = time.perf_counter()-start
    prompt = ids[:, :min(512,ids.shape[1])]
    decode_steps = 64 if steps >= 30 else 4
    decode_times = []
    with torch.inference_mode():
        for repeat in range(4):
            result = model(input_ids=prompt,use_cache=True,output_router_logits=False,
                           logits_to_keep=1,return_dict=True)
            cache = result.past_key_values
            next_ids = result.logits[:,-1:].argmax(-1)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(decode_steps):
                result = model(input_ids=next_ids,past_key_values=cache,use_cache=True,
                               output_router_logits=False,logits_to_keep=1,return_dict=True)
                cache = result.past_key_values
                next_ids = result.logits[:,-1:].argmax(-1)
            torch.cuda.synchronize(device)
            if repeat:
                decode_times.append((time.perf_counter()-start)*1000/decode_steps)
            del result,cache
    return {'kind': 'batch=1 full sequence prefill; no routing hooks/CE',
            'steps': steps, 'sequence_length': ids.shape[1], 'input_tokens_per_second': ids.numel()*steps/seconds,
            'milliseconds_per_forward': seconds*1000/steps,
            'decode_milliseconds_per_token': float(np.mean(decode_times)),
            'decode_repeats_ms': decode_times, 'decode_prompt_tokens': prompt.shape[1],
            'decode_steps': decode_steps,
            'decode_note': 'KV-cache greedy decode, forced fixed length ignoring EOS; 1 warmup + 3 measured repeats',
            'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/GIB}


def gradient_scores(cfg, key, device, output):
    """Squared norm of MEAN LM gradient; greedy cap-three selection is done later.
    CPU FP32 accumulated expert gradients. No optimizer update or test access.
    """
    meta_path = output / 'gradient_scores.json'
    if meta_path.exists():
        return bu.load_json(meta_path)
    seed_all(cfg['seeds'][0])
    torch.set_num_threads(cfg['cpu_threads'])
    model = load_model(cfg, key, device)
    blocks = bu.iter_moe_blocks(model)
    for _, moe, _, _ in blocks:
        for e in moe.experts:
            e.requires_grad_(True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    calib = data_for(cfg, key, 'calibration')
    desc = Path(cfg['models'][key]['descriptors'])
    indices = np.load(desc / 'fit_calibration_indices.npy', allow_pickle=False)[:cfg['gradient_batches']]
    accum = {name: torch.zeros_like(p, dtype=torch.float32, device='cpu')
             for name, p in model.named_parameters() if p.requires_grad}
    t0 = time.perf_counter()
    for j, ix in enumerate(indices):
        ids = bu.make_batch(calib, int(ix), int(ix)+1, device)
        result = bu.forward(model, ids)
        loss = lm_loss(result.logits, ids)
        loss.backward()
        for name, p in model.named_parameters():
            if name in accum and p.grad is not None:
                accum[name].add_(p.grad.detach().float().cpu(), alpha=1/len(indices))
                p.grad = None
        del loss, result
        print(f'EU gradient mean: {j+1}/{len(indices)} batches; no weight updates.', flush=True)
    scores = []
    for idx, moe, n, _ in blocks:
        values = []
        for e in range(n):
            prefix = f'model.layers.{idx}.mlp.experts.{e}.'
            values.append(sum(v.double().square().sum().item() for name, v in accum.items() if name.startswith(prefix)))
        scores.append({'layer': idx, 'scores': values})
    report = {'layers': scores, 'seconds': time.perf_counter()-t0,
              'fit_calibration_indices': [int(x) for x in indices],
              'definition': 'sum of squared FP32 accumulated mean LM parameter gradients per expert',
              'cap_added_copies_per_expert': 3, 'new_router_bias_uniform': [-.001, .001],
              'adaptation': 'native LB, 1.5x growth and new+router-only CPT; not a full paper replication'}
    bu.save_json(meta_path, report)
    del accum, model, blocks
    gc.collect()
    torch.cuda.empty_cache()
    return report


def experiment_recipe(cfg, key, method, seed, plan):
    import transformers
    return {'format': 'graftmoe-v1', 'model': key, 'method': method, 'seed': seed,
            'config': cfg, 'plan': plan, 'torch': str(torch.__version__), 'transformers': transformers.__version__,
            'code_sha256': {p.name: bu.sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'base_checkpoint': bu.checkpoint_identity(Path(cfg['models'][key]['path'])),
            'data_report_sha256': bu.sha256(Path(cfg['models'][key]['data'])/'report.json'),
            'descriptor_report_sha256': (bu.sha256(Path(cfg['models'][key]['descriptors'])/'report.json')
                                         if method not in ('original', 'continue') else None)}


def save_delta(model, out, method, old):
    directory = out / 'final_delta'
    directory.mkdir(exist_ok=True)
    files = {}
    for idx, moe, _, _ in bu.iter_moe_blocks(model):
        values = {'gate.'+n: p.detach().cpu() for n, p in moe.gate.named_parameters()}
        for e in range(old, len(moe.experts)):
            values.update({f'experts.{e}.'+n: p.detach().cpu() for n, p in moe.experts[e].named_parameters()})
        path = directory / f'layer_{idx:02d}.pt'
        atomic_torch(path, values)
        files[path.name] = bu.sha256(path)
    bu.save_json(directory/'manifest.json', {'files': files, 'method': method, 'old_experts': old,
                                            'note': 'Combine with exact base checkpoint and plan.json; no optimizer.'})


def save_full_model(model, source_path, out, method):
    """Save continued-training weights when every model parameter was updated."""
    from transformers import AutoTokenizer
    directory = out / 'final_model'
    directory.mkdir(exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True, max_shard_size='5GB')
    AutoTokenizer.from_pretrained(source_path, local_files_only=True).save_pretrained(directory)
    files = {p.name: bu.sha256(p) for p in sorted(directory.iterdir())
             if p.is_file() and p.name != 'manifest.json'}
    bu.save_json(directory/'manifest.json', {'format': 'hf-pretrained', 'method': method,
                                            'files': files})


def run_experiment(cfg, key, method, seed, out, plan, device):
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process is already using {out}.')
        return _run_experiment(cfg, key, method, seed, out, plan, device)


def _run_experiment(cfg, key, method, seed, out, plan, device):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        print(f'Already complete: {out}', flush=True)
        return
    seed_all(seed)
    torch.set_num_threads(cfg['cpu_threads'])
    recipe = experiment_recipe(cfg, key, method, seed, plan)
    recipe_hash = digest(recipe)
    if (out/'recipe.json').exists() and digest(bu.load_json(out/'recipe.json')) != recipe_hash:
        raise ValueError('Existing run has a different recipe. Do not mix protocols in one directory.')
    bu.save_json(out/'recipe.json', recipe)
    recover_manifest(out, recipe_hash)
    if plan:
        bu.save_json(out/'plan.json', plan)
    valid = data_for(cfg, key, 'valid')
    train = data_for(cfg, key, 'train')
    old = cfg['models'][key]['old_experts']
    model = load_model(cfg, key, device)
    quick_n = cfg['quick_valid_blocks']
    quick = np.sort(np.random.default_rng(1701).choice(len(valid), quick_n, replace=False))
    full_valid = np.arange(len(valid))
    if not (out/'source_quick.json').exists():
        evaluate(model, valid, quick, out, 'source_quick', old, device)
    if method not in ('continue', 'original'):
        apply_expansion(model, plan, .001 if method == 'eu_gn' else 0., seed)
        if method in CIRA_METHODS:
            cira.install(model, plan, cfg['routing_tau'])
    total_params = sum(p.numel() for p in model.parameters())
    warm_seconds = 0.
    if not (out/'latest.json').exists() and method not in ('continue', 'original'):
        if not (out/'after_copy.json').exists():
            evaluate(model, valid, quick, out, 'after_copy', old, device)
        if method in WARM_METHODS:
            warm_seconds = warm_start(model, plan, cfg['models'][key]['descriptors'], out, cfg, seed, device)
    if (out/'warm_report.json').exists():
        warm_seconds = bu.load_json(out/'warm_report.json')['completed_layer_seconds']
    groups = parameter_groups(model, method, old, cfg['learning_rate'])
    num_trainable = sum(p.numel() for _, entries, _ in groups for _, p in entries)
    train_seconds, last_step = 0., 0
    if method != 'original':
        optimizer = MasterAdafactor(groups, cfg['weight_decay'])
        start_step, train_seconds = restore_checkpoint(out, optimizer, recipe_hash, device)
        last_step = start_step
        if not start_step:
            evaluate(model, valid, quick, out, 'valid_step00000', old, device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        total_steps = cfg['steps']
        accumulation = cfg['accumulation']
        need = total_steps * accumulation
        if need > len(train):
            raise ValueError('Budget exceeds one pass of train.npy; change protocol explicitly for multiple epochs.')
        order = np.random.default_rng(seed).permutation(len(train))[:need]
        atomic_array(out/'train_order.npy', order)
        training_peak = 0.
        for step in range(start_step+1, total_steps+1):
            model.train()
            exploration = cira.set_progress(model, step, total_steps) if method in CIRA_METHODS else 0.
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            loss_value = balance_value = 0.
            for ix in order[(step-1)*accumulation:step*accumulation]:
                ids = bu.make_batch(train, int(ix), int(ix)+1, device)
                result = model(input_ids=ids, use_cache=False, output_router_logits=True, return_dict=True)
                lm = lm_loss(result.logits, ids)
                lb = balance_loss(result.router_logits, bu.iter_moe_blocks(model))
                loss = (lm + cfg['balance_weight']*lb)/accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss.')
                loss.backward()
                loss_value += lm.item()/accumulation
                balance_value += lb.item()/accumulation
                del result, lm, lb, loss
            norm = optimizer.step(lr_multiplier(step, total_steps, cfg['lr_warmup_steps']))
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-start
            train_seconds += elapsed
            peak = torch.cuda.max_memory_allocated(device)/GIB
            training_peak = max(training_peak, peak)
            row = {'step': step, 'input_tokens': step*accumulation*train.shape[1],
                   'exploration_probability': exploration, 'train_nll': loss_value, 'lb': balance_value, 'grad_norm': norm,
                   'train_seconds': train_seconds, 'seconds_step': elapsed, 'peak_allocated_gib': peak}
            with (out/'train.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
                f.flush()
            if step == 1 or step % cfg['log_every'] == 0:
                eta = elapsed*(total_steps-step)/3600
                print(f'{key}/{method}/seed{seed}: {step}/{total_steps} NLL={loss_value:.6f} '
                      f'LB={balance_value:.4f} peak={peak:.2f} GiB ETA~{eta:.1f} h', flush=True)
            if step % cfg['eval_every'] == 0 or step == total_steps:
                evaluate(model, valid, quick, out, f'valid_step{step:05d}', old, device, step, train_seconds)
            if step % cfg['save_every'] == 0 or step == total_steps:
                save_checkpoint(out, optimizer, step, recipe_hash, train_seconds, device)
            last_step = step
        del optimizer, groups
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    model.requires_grad_(False).eval()
    vf = evaluate(model, valid, full_valid, out, 'valid_final', old, device, last_step, train_seconds)
    test = data_for(cfg, key, 'test')
    test_indices = np.arange(len(test))
    tf = evaluate(model, test, test_indices, out, 'test_final', old, device, last_step, train_seconds)
    tasks = run_tasks(model, key, cfg, out, device) if cfg['tasks'] else {}
    bench = inference_benchmark(model, valid, device, cfg['benchmark_steps']) if cfg['benchmark_steps'] else None
    bu.save_json(out/'benchmark.json', bench)
    if cfg['save_final_weights']:
        if method not in ('original', 'continue'):
            save_delta(model, out, method, old)
        elif method == 'continue':
            save_full_model(model, cfg['models'][key]['path'], out, method)
    report = {'complete': True, 'model': key, 'method': method, 'seed': seed,
              'final_step': last_step, 'input_tokens': last_step*cfg['accumulation']*train.shape[1],
              'prediction_tokens': last_step*cfg['accumulation']*(train.shape[1]-1),
              'num_parameters': total_params, 'trainable_parameters': num_trainable,
              'train_seconds': train_seconds, 'warm_seconds': warm_seconds,
              'validation': vf, 'test': tf, 'tasks': {k: {'score': v['score'], 'metric': v['metric']} for k,v in tasks.items()},
              'benchmark': bench, 'recipe_hash': recipe_hash,
              'cost_note': 'Single-GPU CPU-offloaded Adafactor; training excludes validation/checkpoint I/O; '
                           'concurrent CPU/I/O jobs can change timing. No distributed communication or FLOP claim.'}
    # Durable metrics first; cleanup only this completed run's optimizer snapshots.
    bu.save_json(out/'complete.json', report)
    if (out/'warm_cache').exists():
        shutil.rmtree(out/'warm_cache')
    if not cfg['keep_optimizer_checkpoints']:
        cleanup_completed_run(out)
    print(f'COMPLETE: {out}', flush=True)
