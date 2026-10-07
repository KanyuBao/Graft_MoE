#!/usr/bin/env python3
"""Collect GRAFT-MoE descriptors from local calibration data and native HF MoE layers.

Only calibration.npy is read. Calibration blocks are split before any statistics
or probes are collected. Expert inputs are observed through forward hooks; the
checkpoint's forward and routing are never replaced. Models run sequentially.

  python -u collect_descriptors.py --config protocol.json --models olmoe qwen --device cuda:0
"""

import argparse
from contextlib import contextmanager
import gc
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

import baseline_utils as bu
from project_config import load_config

HERE = Path(__file__).resolve().parent
CALIBRATION_BLOCKS = 512
SEQUENCE_LENGTH = 2048


def atomic_tensor(path, value):
    import torch
    temporary = path.with_name(path.name + '.building')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_array(path, value):
    temporary = path.with_name(path.name + '.building')
    with temporary.open('wb') as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def collection_settings(cfg):
    seed = int(cfg.get('descriptor_seed', 42))
    fraction = float(cfg.get('descriptor_fit_fraction', .75))
    probes = int(cfg.get('descriptor_probe_count', 256))
    dimension = int(cfg.get('descriptor_projection_dim', 128))
    batch = int(cfg.get('descriptor_probe_batch', 128))
    if seed < 0 or not 0 < fraction < 1 or min(probes, dimension, batch) < 1:
        raise ValueError('Descriptor seed must be nonnegative; fraction in (0,1); sizes positive.')
    fit_blocks = int(CALIBRATION_BLOCKS * fraction)
    if not 0 < fit_blocks < CALIBRATION_BLOCKS:
        raise ValueError('Both calibration partitions must contain at least one whole block.')
    if probes > min(fit_blocks, CALIBRATION_BLOCKS - fit_blocks) * SEQUENCE_LENGTH:
        raise ValueError('Probe count exceeds a calibration partition; sampling uses no replacement.')
    return dict(seed=seed, fit_fraction=fraction, fit_blocks=fit_blocks,
                probe_count=probes, projection_dim=dimension, probe_batch=batch)


def partition_calibration(settings):
    permutation = np.random.default_rng(settings['seed']).permutation(CALIBRATION_BLOCKS)
    cut = settings['fit_blocks']
    indices = {'fit': np.sort(permutation[:cut]), 'check': np.sort(permutation[cut:])}
    positions = {}
    for phase, offset in (('fit', 100003), ('check', 200003)):
        tokens = len(indices[phase]) * SEQUENCE_LENGTH
        positions[phase] = np.sort(np.random.default_rng(settings['seed'] + offset).choice(
            tokens, settings['probe_count'], replace=False))
    return indices, positions


@contextmanager
def output_directory(path):
    import fcntl
    path.mkdir(parents=True, exist_ok=True)
    with (path / 'collection.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another descriptor collection is using {path}.') from exc
        if any(p.name != 'collection.lock' for p in path.iterdir()):
            raise FileExistsError(f'Descriptor output is not empty: {path}. Choose a new directory.')
        yield


class LayerCollector:
    """Merge per-invocation moments on CPU; retain only a fixed small probe bank.

    GPU temporaries have shape experts x hidden_size, never calibration_tokens x
    hidden_size. A native expert invocation supplies the actual routed inputs.
    Global softmax probabilities are conditioned on that expert's native top-k
    membership, before optional selected-weight renormalization.
    """

    def __init__(self, index, moe, n, k, width, positions):
        import torch
        self.torch = torch
        self.index, self.moe, self.n, self.k, self.width = index, moe, n, k, width
        self.positions = positions
        self.handles = []
        self.phase = None
        self.offset = 0
        self.pending = False
        self.values = {}
        for phase in ('fit', 'check'):
            self.values[phase] = {
                'counts': torch.zeros(n, dtype=torch.int64),
                'mean': torch.zeros(n, width, dtype=torch.float64),
                'm2': torch.zeros(n, width, dtype=torch.float64),
                'probability_sum': torch.zeros(n, dtype=torch.float64),
                'hidden': torch.empty(len(positions[phase]), width, dtype=moe.gate.weight.dtype),
                'probe_filled': np.zeros(len(positions[phase]), dtype=bool),
                'input_tokens': 0, 'shared_count': 0,
            }
        device = moe.gate.weight.device
        self.batch_mean = torch.zeros(n, width, dtype=torch.float32, device=device)
        self.batch_m2 = torch.zeros_like(self.batch_mean)

    def set_batch(self, phase, offset):
        if self.pending:
            raise RuntimeError(f'Layer {self.index}: previous forward did not complete.')
        self.phase, self.offset = phase, offset

    def record_gate(self, module, inputs, logits):
        torch = self.torch
        if self.phase is None or self.pending:
            raise RuntimeError(f'Layer {self.index}: unexpected gate invocation.')
        hidden = inputs[0].detach().reshape(-1, self.width)
        if logits.ndim != 2 or tuple(logits.shape) != (len(hidden), self.n):
            raise ValueError(f'Layer {self.index}: unsupported native gate output shape.')
        if len(hidden) != SEQUENCE_LENGTH:
            raise ValueError('Collection requires one complete packed calibration block per forward.')
        self.pending = True
        self.batch_mean.zero_()
        self.batch_m2.zero_()
        self.actual_counts = [0] * self.n
        self.seen = [False] * self.n
        self.shared_count = 0
        self.batch_tokens = len(hidden)
        probabilities = logits.detach().float().softmax(-1)
        selected = probabilities.topk(self.k, dim=-1).indices
        self.expected_counts = torch.bincount(selected.reshape(-1), minlength=self.n)
        self.probability_sum = torch.zeros(self.n, device=logits.device, dtype=torch.float32)
        self.probability_sum.scatter_add_(0, selected.reshape(-1),
                                         probabilities.gather(1, selected).reshape(-1))
        positions = self.positions[self.phase]
        first = int(np.searchsorted(positions, self.offset))
        last = int(np.searchsorted(positions, self.offset + len(hidden)))
        if first != last:
            local = torch.tensor(positions[first:last] - self.offset,
                                 dtype=torch.long, device=hidden.device)
            values = self.values[self.phase]
            values['hidden'][first:last].copy_(hidden.index_select(0, local).cpu())
            values['probe_filled'][first:last] = True

    def record_expert(self, expert_id, inputs):
        if not self.pending:
            raise RuntimeError(f'Layer {self.index}: expert invoked before gate.')
        if self.seen[expert_id]:
            raise ValueError('Native expert is invoked more than once per block; unsupported dispatch.')
        self.seen[expert_id] = True
        hidden = inputs[0].detach().reshape(-1, self.width).float()
        count = len(hidden)
        self.actual_counts[expert_id] = count
        if count:
            mean = hidden.mean(0)
            self.batch_mean[expert_id].copy_(mean)
            self.batch_m2[expert_id].copy_((hidden - mean).square().sum(0))

    def record_shared(self, module, inputs):
        if not self.pending:
            raise RuntimeError(f'Layer {self.index}: shared expert invoked before gate.')
        self.shared_count += inputs[0].numel() // self.width

    def finish_layer(self, module, inputs, output):
        torch = self.torch
        if not self.pending:
            raise RuntimeError(f'Layer {self.index}: native gate hook was not called.')
        counts = torch.tensor(self.actual_counts, dtype=torch.int64)
        if not torch.equal(counts, self.expected_counts.cpu()) or int(counts.sum()) != self.batch_tokens * self.k:
            raise ValueError(f'Layer {self.index}: actual expert dispatch differs from native global top-k.')
        expected_shared = self.batch_tokens if hasattr(self.moe, 'shared_expert') else 0
        if self.shared_count != expected_shared:
            raise ValueError(f'Layer {self.index}: shared expert invocation count mismatch.')
        values = self.values[self.phase]
        previous = values['counts'].double().unsqueeze(1)
        added = counts.double().unsqueeze(1)
        total = (previous + added).clamp_min(1)
        delta = self.batch_mean.cpu().double() - values['mean']
        values['mean'].add_(delta * added / total)
        values['m2'].add_(self.batch_m2.cpu().double() + delta.square() * previous * added / total)
        values['counts'].add_(counts)
        values['probability_sum'].add_(self.probability_sum.cpu().double())
        values['input_tokens'] += self.batch_tokens
        values['shared_count'] += self.shared_count
        self.pending = False

    def attach(self):
        self.handles.append(self.moe.gate.register_forward_hook(self.record_gate))
        for expert_id, expert in enumerate(self.moe.experts):
            def record(module, inputs, e=expert_id):
                self.record_expert(e, inputs)
            self.handles.append(expert.register_forward_pre_hook(record))
        if hasattr(self.moe, 'shared_expert'):
            self.handles.append(self.moe.shared_expert.register_forward_pre_hook(self.record_shared))
        self.handles.append(self.moe.register_forward_hook(self.finish_layer))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def partition_payload(self, phase, block_indices):
        torch = self.torch
        values = self.values[phase]
        counts = values['counts']
        expected_tokens = len(block_indices) * SEQUENCE_LENGTH
        if values['input_tokens'] != expected_tokens or int(counts.sum()) != expected_tokens * self.k:
            raise ValueError(f'Layer {self.index}/{phase}: incomplete calibration statistics.')
        if not values['probe_filled'].all():
            raise ValueError(f'Layer {self.index}/{phase}: incomplete probe bank.')
        if not bool((counts > 0).all()):
            missing = torch.where(counts == 0)[0].tolist()
            raise ValueError(f'Layer {self.index}/{phase}: unobserved experts {missing}; '
                             'no descriptor imputation is performed. Review calibration coverage.')
        denominator = counts.double().clamp_min(1)
        positions = self.positions[phase]
        coordinates = np.stack((block_indices[positions // SEQUENCE_LENGTH], positions % SEQUENCE_LENGTH), axis=1)
        result = {
            'counts': counts, 'observed': counts > 0,
            'mu': values['mean'].float(),
            'sigma': (values['m2'] / denominator[:, None]).clamp_min(0).sqrt().float(),
            'a': (values['probability_sum'] / denominator).float(),
            'hidden': values['hidden'],
            'probe_calibration_coordinates': torch.tensor(coordinates, dtype=torch.int64),
            'input_tokens': values['input_tokens'], 'shared_count': values['shared_count'],
        }
        for name in ('mu', 'sigma', 'a', 'hidden'):
            if not bool(torch.isfinite(result[name]).all()):
                raise FloatingPointError(f'Layer {self.index}/{phase}/{name}: non-finite descriptor.')
        return result


def projected_responses(moe, hidden, projection, batch_size):
    import torch
    n, probes = len(moe.experts), len(hidden)
    result = torch.empty(n, probes, projection.shape[1], dtype=torch.float32)
    device = moe.gate.weight.device
    # Every routed expert sees the SAME ordered probes and SAME projection.
    for expert_id, expert in enumerate(moe.experts):
        for start in range(0, probes, batch_size):
            end = min(probes, start + batch_size)
            output = expert(hidden[start:end].to(device=device, dtype=moe.gate.weight.dtype))
            if output.shape != (end - start, projection.shape[0]):
                raise ValueError('Native routed expert output shape is incompatible with common probes.')
            result[expert_id, start:end].copy_((output.float() @ projection).cpu())
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Non-finite common-probe response.')
    return result


def collect_model(cfg, key, device, config_path):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    started = time.perf_counter()
    settings = collection_settings(cfg)
    spec = cfg['models'][key]
    model_path, output = Path(spec['path']), Path(spec['descriptors'])
    args = SimpleNamespace(data_dir=Path(spec['data']), split='calibration', model=key,
                           model_path=model_path, max_sequences=0)
    data, _, data_info = bu.validate_data(args)
    if tuple(data.shape) != (CALIBRATION_BLOCKS, SEQUENCE_LENGTH):
        raise ValueError('Descriptor calibration must contain exactly 512 packed blocks of length 2048.')
    data_info['selection'] = 'All calibration blocks; block-disjoint random fit/check partition saved as index arrays.'
    native_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    expected_type = 'olmoe' if key == 'olmoe' else 'qwen2_moe'
    if native_config.model_type != expected_type:
        raise ValueError(f'{key}: expected native model_type={expected_type}.')
    if int(data.min()) < 0 or int(data.max()) >= native_config.vocab_size:
        raise ValueError('Calibration token IDs are outside the checkpoint vocabulary.')
    identity = bu.checkpoint_identity(model_path)
    identity['weight_sha256'] = {row['name']: bu.sha256(model_path / row['name'])
                               for row in identity['weight_files']}
    identity['weight_identity_note'] = 'Full SHA256 plus byte size and modification time for every weight shard.'
    indices, positions = partition_calibration(settings)
    report = {
        'format_version': 2, 'complete': False,
        'arguments': {'model': key, 'config': str(config_path), 'device': str(device), **settings},
        'execution_dtype': 'bfloat16',
        'checkpoint': identity, 'model_config': native_config.to_dict(), 'data': data_info,
        'calibration_shape': [CALIBRATION_BLOCKS, SEQUENCE_LENGTH],
        'partition_input_tokens': {p: len(v) * SEQUENCE_LENGTH for p, v in indices.items()},
        'partition_blocks': {p: len(v) for p, v in indices.items()},
        'partition_rule': 'Seeded block permutation; disjoint fit/check; all 512 calibration blocks used once.',
        'files_sha256': {}, 'layers': [],
        'provenance_seconds': time.perf_counter() - started,
        'definitions': {
            'mu_sigma': 'Per-expert routed-input mean and population standard deviation; '
                        'actual native expert invocations; float32 block moments merged with float64 parallel Welford.',
            'a': 'Mean full-expert global softmax p(e|x), conditional on native top-k dispatch to e; '
                 'before selected-weight renormalization; not cluster-conditional probability.',
            'responses': 'Unweighted routed-expert outputs on the same ordered layer-input probes, '
                         'multiplied by one Gaussian projection shared by all experts, layers and both partitions.',
            'probes': 'Uniform token sampling without replacement within each block-disjoint partition; '
                      'fixed before the model is run; no selection on held-out responses.',
            'shared_expert': 'Excluded from routed descriptors and responses; native invocation counts recorded separately.',
            'hidden': 'The original native gate input at selected calibration token coordinates; reused by layer warm-start.',
        },
        'numpy_version': np.__version__, 'torch_version': str(torch.__version__),
        'transformers_version': transformers.__version__, 'python_version': sys.version,
        'script_sha256': bu.sha256(__file__),
    }
    with output_directory(output):
        bu.save_json(output / 'report.json', report)
        collectors = []
        model = None
        try:
            for phase in ('fit', 'check'):
                name = f'{phase}_calibration_indices.npy'
                atomic_array(output / name, indices[phase])
                report['files_sha256'][name] = bu.sha256(output / name)
            torch.manual_seed(settings['seed'])
            torch.set_num_threads(int(cfg.get('cpu_threads', 4)))
            if device.type == 'cuda':
                torch.cuda.set_device(device)
                torch.cuda.reset_peak_memory_stats(device)
            load_begin = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, use_safetensors=True, torch_dtype=torch.bfloat16,
                attn_implementation=cfg.get('attention', 'sdpa'), low_cpu_mem_usage=True)
            model.config.use_cache = False
            model.to(device).eval().requires_grad_(False)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            report['load_seconds'] = time.perf_counter() - load_begin
            width = int(model.config.hidden_size)
            if settings['projection_dim'] > width:
                raise ValueError('Descriptor projection dimension must not exceed hidden_size.')
            blocks = bu.iter_moe_blocks(model)
            for index, moe, n, k in blocks:
                if n != spec['old_experts'] or k != spec['top_k']:
                    raise ValueError('Native expert count/top-k differs from the configured protocol.')
                collector = LayerCollector(index, moe, n, k, width, positions)
                collector.attach()
                collectors.append(collector)
            begin = time.perf_counter()
            with torch.inference_mode():
                for phase in ('fit', 'check'):
                    for row, calibration_index in enumerate(indices[phase]):
                        for collector in collectors:
                            collector.set_batch(phase, row * SEQUENCE_LENGTH)
                        ids = torch.tensor(np.asarray(data[int(calibration_index)]).copy(),
                                           dtype=torch.long, device=device).unsqueeze(0)
                        # The native decoder suffices: skip the unused vocabulary projection.
                        result = model.model(input_ids=ids, use_cache=False, output_router_logits=False,
                                             output_hidden_states=False, output_attentions=False, return_dict=True)
                        del result, ids
                        if (row + 1) % 32 == 0 or row + 1 == len(indices[phase]):
                            print(f'{key}/{phase}: {row + 1}/{len(indices[phase])} calibration blocks', flush=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            report['calibration_seconds'] = time.perf_counter() - begin
            for collector in collectors:
                collector.remove()
            begin = time.perf_counter()
            projection_seed = settings['seed'] + 300007
            generator = torch.Generator(device='cpu').manual_seed(projection_seed)
            projection_cpu = torch.randn(width, settings['projection_dim'], generator=generator) / math.sqrt(settings['projection_dim'])
            atomic_tensor(output / 'projection.pt', projection_cpu)
            report['files_sha256']['projection.pt'] = bu.sha256(output / 'projection.pt')
            report['response_projection'] = {'file': 'projection.pt', 'seed': projection_seed,
                                             'shape': list(projection_cpu.shape), 'distribution': 'iid N(0, 1/output_dim)'}
            projection = projection_cpu.to(device)
            with torch.inference_mode():
                for collector in collectors:
                    payload = {'format_version': 2, 'layer': collector.index,
                               'num_experts': collector.n, 'top_k': collector.k,
                               'norm_topk_prob': bool(getattr(collector.moe, 'norm_topk_prob', False)),
                               'has_shared_expert': hasattr(collector.moe, 'shared_expert'),
                               'projection_file': 'projection.pt'}
                    for phase in ('fit', 'check'):
                        values = collector.partition_payload(phase, indices[phase])
                        values['responses'] = projected_responses(collector.moe, values['hidden'],
                                                                   projection, settings['probe_batch'])
                        payload[phase] = values
                    name = f'layer_{collector.index:02d}.pt'
                    atomic_tensor(output / name, payload)
                    report['files_sha256'][name] = bu.sha256(output / name)
                    report['layers'].append({k: payload[k] for k in ('layer', 'num_experts', 'top_k',
                                                                     'norm_topk_prob', 'has_shared_expert')})
                    collector.values.clear()
                    print(f'{key}: saved {name}; common probes={settings["probe_count"]}, '
                          f'projection={settings["projection_dim"]}', flush=True)
                    del payload, values
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            report['probe_and_save_seconds'] = time.perf_counter() - begin
            # Detect input replacement during collection before committing a usable report.
            current = bu.checkpoint_identity(model_path)
            if any(current[k] != identity[k] for k in ('config_sha256', 'index_sha256', 'weight_files')):
                raise ValueError('Checkpoint files changed during descriptor collection.')
            if bu.sha256(Path(data_info['path'])) != data_info['sha256']:
                raise ValueError('Calibration data changed during descriptor collection.')
            if bu.sha256(Path(spec['data']) / 'report.json') != data_info['preparation_report_sha256']:
                raise ValueError('Calibration provenance report changed during descriptor collection.')
            report['peak_allocated_gib'] = (torch.cuda.max_memory_allocated(device) / 1024**3
                                            if device.type == 'cuda' else None)
            report['elapsed_seconds'] = time.perf_counter() - started
            report['complete'] = True
            bu.save_json(output / 'report.json', report)
            print(f'{key}: complete descriptors at {output}; {report["elapsed_seconds"]:.1f}s', flush=True)
        except Exception as exc:
            report['complete'] = False
            report['error_type'], report['error'] = type(exc).__name__, str(exc)
            bu.save_json(output / 'report.json', report)
            raise
        finally:
            for collector in collectors:
                collector.remove()
            collectors.clear()
            model = None
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=HERE / 'protocol.json')
    parser.add_argument('--models', nargs='+', choices=('olmoe', 'qwen'), default=['olmoe', 'qwen'])
    parser.add_argument('--device', default='cuda:0', help='One resident model at a time; default cuda:0.')
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error('--models must not contain duplicates.')
    return args


def main():
    args = parse_args()
    import torch
    cfg = load_config(args.config)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and single-device CUDA collection are supported.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use the configured PyTorch/CUDA environment.')
    for key in args.models:
        collect_model(cfg, key, device, args.config.expanduser().resolve())


if __name__ == '__main__':
    main()
