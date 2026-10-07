#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU group analysis and expansion plans for collect_descriptors.py.

Requires Python 3.11+, NumPy and PyTorch. No sklearn/SciPy/Transformers/model
weights/GPU needed. Reads existing layer_XX.pt and verifies their SHA256.
Plans depend ONLY on fit statistics and responses. Check responses measure
held-out similarity; all seeds are retained. No automatic method/seed selection.

K-means is ordinary Euclidean Lloyd with D^2 initialization (multiple starts).
The N-by-N linear Gram matrix evaluates distances to arithmetic centroids
exactly; it introduces no nonlinear kernel, projection or dimension reduction.

  python -u group_utils.py --descriptor-dir /path/to/olmoe_descriptors --expand-to 96
  python -u group_utils.py --descriptor-dir /path/to/qwen_descriptors --expand-to 90

Main plans: full, random_groups, random_copy, traffic_top.
Descriptor ablations: no_router, input_only, response_only.
These are controlled initialization plans, not reproductions of published
Expert Upcycling / Orthogonal Growth training methods.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

# This CPU stage uses small expert-by-expert matrices. Explicit shell settings
# take precedence. Set before NumPy/PyTorch initialize their math libraries.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '2')
import numpy as np

VARIANTS = {
    'full': ('mu', 'sigma', 'responses', 'a'),
    'no_router': ('mu', 'sigma', 'responses'),
    'input_only': ('mu', 'sigma'),
    'response_only': ('responses',),
}
METHODS = (*VARIANTS, 'random_groups', 'random_copy', 'traffic_top')


def log(message):
    print(message, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def array(tensor):
    # Avoid torch 2.3 <-> NumPy 2 binary bridge incompatibility.
    return np.asarray(tensor.detach().cpu().tolist(), dtype=np.float64)


def normalized_blocks(fit):
    result, metadata = {}, {}
    n = len(fit['counts'])
    for name in ('mu', 'sigma', 'responses', 'a'):
        raw = array(fit[name]).reshape(n, -1)
        if not np.isfinite(raw).all():
            raise ValueError(f'fit.{name} 存在 NaN/Inf，不能聚类。')
        centered = raw - raw.mean(axis=0)
        energy = float(np.mean(np.sum(centered**2, axis=1)))
        raw_energy = float(np.mean(np.sum(raw**2, axis=1)))
        active = energy > max(1e-20, 1e-12 * raw_energy)
        scale = math.sqrt(energy) if active else 0.0
        result[name] = centered / scale if active else np.zeros_like(centered)
        metadata[name] = {'dim': raw.shape[1], 'centered_energy': energy,
                          'raw_energy': raw_energy, 'scale': scale, 'active': active}
    return result, metadata


def distance_matrix(gram):
    diagonal = np.diag(gram)
    result = np.maximum(diagonal[:, None] + diagonal[None, :] - 2 * gram, 0)
    np.fill_diagonal(result, 0)
    return result


def canonical_labels(labels):
    groups = sorted(np.unique(labels), key=lambda c: int(np.flatnonzero(labels == c)[0]))
    mapping = {int(old): new for new, old in enumerate(groups)}
    return np.asarray([mapping[int(v)] for v in labels], dtype=np.int64)


def centroid_distances(gram, labels, k):
    assignment = np.eye(k, dtype=np.float64)[labels]
    sizes = assignment.sum(axis=0)
    if (sizes == 0).any():
        raise ValueError('不能计算空簇中心。')
    average_weights = assignment / sizes
    cross = gram @ average_weights
    center_norms = np.sum(average_weights * cross, axis=0)
    return np.maximum(np.diag(gram)[:, None] - 2 * cross + center_norms, 0)


def fill_empty(labels, distances, k):
    labels = labels.copy()
    sizes = np.bincount(labels, minlength=k)
    for empty in np.flatnonzero(sizes == 0):
        candidates = np.flatnonzero(sizes[labels] > 1)
        if not len(candidates):
            raise RuntimeError('无法补齐非空簇。')
        # Move the point furthest from its currently assigned center.
        error = distances[candidates, labels[candidates]]
        donor = int(candidates[np.argmax(error)])
        sizes[labels[donor]] -= 1
        labels[donor] = empty
        sizes[empty] += 1
    return labels


def fit_kmeans(gram, k, seed, n_init=10, max_iter=300):
    n = len(gram)
    if not 1 < k < n:
        raise ValueError('K 必须大于 1 且小于专家数。')
    pairwise = distance_matrix(gram)
    rng = np.random.default_rng(seed)
    best = None
    for initialization in range(n_init):
        centers = [int(rng.integers(n))]
        closest = pairwise[:, centers[0]].copy()
        while len(centers) < k:
            closest[centers] = 0
            total = float(closest.sum())
            if total <= 1e-14:
                raise ValueError('有效的不同描述符少于 K，请检查特征或降低 K。')
            selected = int(rng.choice(n, p=closest / total))
            centers.append(selected)
            closest = np.minimum(closest, pairwise[:, selected])
        distances = pairwise[:, centers]
        labels = fill_empty(distances.argmin(axis=1), distances, k)
        converged = False
        for iteration in range(1, max_iter + 1):
            distances = centroid_distances(gram, labels, k)
            updated = fill_empty(distances.argmin(axis=1), distances, k)
            if np.array_equal(updated, labels):
                converged = True
                break
            labels = updated
        if not converged:
            raise RuntimeError('K-means 未在 max-iter 内收敛，停止发布计划。')
        distances = centroid_distances(gram, labels, k)
        inertia = float(distances[np.arange(n), labels].sum())
        if best is None or inertia < best['inertia']:
            best = {'labels': canonical_labels(labels), 'inertia': inertia,
                    'iterations': iteration, 'initialization': initialization}
    return best


def adjusted_rand(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    if len(a) != len(b):
        raise ValueError('ARI 输入长度不一致。')
    n = len(a)
    if n < 2:
        return 1.0
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    table = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb)
    pairs = lambda values: float(np.sum(values * (values - 1) / 2))
    observed = pairs(table)
    left, right = pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = left * right / (n * (n - 1) / 2)
    maximum = (left + right) / 2
    if abs(maximum - expected) < 1e-14:
        return 1.0
    return float((observed - expected) / (maximum - expected))


def allocate(scores, budget):
    scores = np.asarray(scores, dtype=np.float64)
    if budget < 1 or not np.isfinite(scores).all() or (scores < 0).any() or scores.sum() <= 0:
        raise ValueError('分配预算和流量非法。')
    quota = budget * scores / scores.sum()
    result = np.floor(quota).astype(np.int64)
    remaining = budget - int(result.sum())
    order = np.argsort(-(quota - result), kind='stable')
    result[order[:remaining]] += 1
    if int(result.sum()) != budget:
        raise RuntimeError('分配预算检查失败。')
    return result


def representative_order(gram, members):
    members = np.asarray(members, dtype=np.int64)
    local = gram[np.ix_(members, members)]
    distance_to_mean = np.diag(local) - 2 * local.mean(axis=1) + local.mean()
    chosen = [int(np.argmin(distance_to_mean))]
    distances = distance_matrix(local)
    closest = distances[:, chosen[0]].copy()
    while len(chosen) < len(members):
        closest[chosen] = -np.inf
        point = int(np.argmax(closest))
        chosen.append(point)
        closest = np.minimum(closest, distances[:, point])
    return members[chosen].tolist()


def make_group_plan(gram, counts, labels, budget):
    k = int(labels.max()) + 1
    traffic = np.bincount(labels, weights=counts, minlength=k)
    allocations = allocate(traffic, budget)
    size_allocation = allocate(np.bincount(labels, minlength=k), budget)
    groups, parents = [], []
    for c in range(k):
        members = np.flatnonzero(labels == c)
        order = representative_order(gram, members)
        copies = [order[i % len(order)] for i in range(int(allocations[c]))]
        parents.extend(copies)
        groups.append({'cluster': c, 'members': members.tolist(),
                       'traffic_share': float(traffic[c] / traffic.sum()),
                       'new_experts': int(allocations[c]),
                       'representative_order': order, 'parents': copies})
    return {'parents': parents, 'labels': labels.tolist(), 'groups': groups,
            'traffic_vs_size_reallocation': int(np.abs(allocations - size_allocation).sum() // 2)}


def simple_plan(counts, budget, method, seed):
    n = len(counts)
    if method == 'random_copy':
        order = np.random.default_rng(seed).permutation(n).tolist()
    elif method == 'traffic_top':
        order = np.argsort(-counts, kind='stable').tolist()
    else:
        raise ValueError(method)
    return {'parents': [int(order[i % n]) for i in range(budget)],
            'labels': None, 'groups': None, 'traffic_vs_size_reallocation': None}


def response_distances(responses):
    flat = array(responses).reshape(len(responses), -1)
    if not np.isfinite(flat).all():
        raise ValueError('留出响应包含 NaN/Inf。')
    flat -= flat.mean(axis=0)
    return distance_matrix(flat @ flat.T)


def response_ratio(distances, labels):
    i, j = np.triu_indices(len(labels), 1)
    same = labels[i] == labels[j]
    denominator = float(distances[i, j].mean())
    if not same.any() or denominator <= 1e-20:
        return None
    return float(distances[i[same], j[same]].mean() / denominator)


def random_reference(distances, labels, repeats, seed):
    rng = np.random.default_rng(seed)
    values = [response_ratio(distances, rng.permutation(labels)) for _ in range(repeats)]
    values = [value for value in values if value is not None]
    if not values:
        return {'mean': None, 'q05': None, 'q95': None}
    return {'mean': float(np.mean(values)), 'q05': float(np.quantile(values, .05)),
            'q95': float(np.quantile(values, .95))}


def parent_response_coverage(distances, parents, weights):
    denominator = float(distances[np.triu_indices(len(distances), 1)].mean())
    if denominator <= 1e-20:
        return None
    nearest = distances[:, np.unique(parents)].min(axis=1)
    return float(np.sum(nearest * weights) / weights.sum() / denominator)


def scalar_mean(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_payload(payload, entry, source, partition_indices=None):
    layer, n, k = int(entry['layer']), int(entry['num_experts']), int(entry['top_k'])
    if (payload['layer'], payload['num_experts'], payload['top_k']) != (layer, n, k):
        raise ValueError('层配置与 report.json 不一致。')
    width = int(source['model_config']['hidden_size'])
    for phase in ('fit', 'check'):
        values = payload[phase]
        counts = array(values['counts'])
        if counts.shape != (n,) or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
            raise ValueError('路由计数格式非法。')
        tokens = int(source['partition_input_tokens'][phase])
        if counts.sum() != tokens * k or values['input_tokens'] != tokens:
            raise ValueError('路由总计数与原生 top-k 不一致。')
        observed = array(values['observed']).astype(bool)
        if not np.array_equal(observed, counts > 0) or not observed.all():
            raise ValueError('存在未观察专家，请先处理校准覆盖问题。')
        if tuple(values['mu'].shape) != (n, width) or tuple(values['sigma'].shape) != (n, width):
            raise ValueError('输入统计维度不符。')
        if tuple(values['a'].shape) != (n,) or len(values['responses'].shape) != 3:
            raise ValueError('路由项或响应维度不符。')
        if values['responses'].shape[0] != n:
            raise ValueError('响应专家数不符。')
        if 'hidden' not in values or len(values['hidden'].shape) != 2 or values['hidden'].shape[1] != width:
            raise ValueError('共同 probe hidden 缺失或维度不符，不能用于整层预热。')
        if values['hidden'].shape[0] < 1 or min(values['responses'].shape[1:]) < 1:
            raise ValueError('共同 probe 或响应不能为空。')
        for name in ('mu', 'sigma', 'a', 'hidden', 'responses'):
            if not np.isfinite(array(values[name])).all():
                raise ValueError(f'{phase}.{name} 含 NaN/Inf。')
        if (array(values['sigma']) < 0).any() or ((array(values['a']) < 0) | (array(values['a']) > 1 + 1e-6)).any():
            raise ValueError('标准差或全局条件路由概率范围非法。')
        if int(payload.get('format_version', 1)) >= 2:
            probes = int(source['arguments']['probe_count'])
            dimension = int(source['response_projection']['shape'][1])
            if tuple(values['hidden'].shape) != (probes, width) or tuple(values['responses'].shape) != (n, probes, dimension):
                raise ValueError('共同 probe 或共享投影响应维度与 report 不一致。')
            coordinates = array(values['probe_calibration_coordinates'])
            if coordinates.shape != (probes, 2) or not np.equal(coordinates, np.floor(coordinates)).all():
                raise ValueError('probe 的 calibration token 坐标非法。')
            if len(np.unique(coordinates, axis=0)) != probes:
                raise ValueError('共同 probe token 坐标重复。')
            if partition_indices is not None and not np.isin(coordinates[:, 0], partition_indices[phase]).all():
                raise ValueError('共同 probe 来自错误的 fit/check 分区。')
            if ((coordinates[:, 1] < 0) | (coordinates[:, 1] >= int(source['calibration_shape'][1]))).any():
                raise ValueError('probe token 位置超出序列长度。')
        expected_shared = tokens if payload['has_shared_expert'] else 0
        if values['shared_count'] != expected_shared:
            raise ValueError('共享专家计数不符。')


def run(args, output):
    import torch
    torch.set_num_threads(2)
    source_path = args.descriptor_dir / 'report.json'
    source = read_json(source_path)
    if not source.get('complete'):
        raise ValueError('请使用完整的描述符采集目录。')
    model = source['arguments']['model']
    if model not in ('olmoe', 'qwen'):
        raise ValueError('只支持本次 olmoe/qwen 数据格式。')
    entries = source['layers']
    numbers = {int(row['num_experts']) for row in entries}
    if len(numbers) != 1:
        raise ValueError('此计划生成器要求各层原始专家数一致。')
    old_n = numbers.pop()
    budget = args.expand_to - old_n
    if budget <= 0 or args.clusters >= old_n:
        raise ValueError('expand-to 必须大于原始专家数，K 必须小于原始专家数。')
    layer_ids = [int(row['layer']) for row in entries]
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError('report 中存在重复层编号。')
    for phase in ('fit', 'check'):
        index_file = f'{phase}_calibration_indices.npy'
        if sha256(args.descriptor_dir / index_file) != source['files_sha256'][index_file]:
            raise ValueError('fit/check 索引 SHA256 不符。')
    fit_indices = np.load(args.descriptor_dir / 'fit_calibration_indices.npy', allow_pickle=False)
    check_indices = np.load(args.descriptor_dir / 'check_calibration_indices.npy', allow_pickle=False)
    for phase, indices in (('fit', fit_indices), ('check', check_indices)):
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices) or (indices < 0).any():
            raise ValueError('fit/check 索引必须是非空的一维非负整数数组。')
        if len(np.unique(indices)) != len(indices):
            raise ValueError('fit/check 分区内部存在重复序列块。')
    if np.intersect1d(fit_indices, check_indices).size:
        raise ValueError('fit 与 check 存在重叠序列块。')
    if int(source.get('format_version', 1)) >= 2:
        count, length = map(int, source['calibration_shape'])
        if not np.array_equal(np.sort(np.concatenate((fit_indices, check_indices))), np.arange(count)):
            raise ValueError('fit/check 没有完整划分 calibration 数据。')
        for phase, indices in (('fit', fit_indices), ('check', check_indices)):
            if len(indices) * length != source['partition_input_tokens'][phase]:
                raise ValueError('分区索引与输入 token 数不一致。')
        projection_name = source['response_projection']['file']
        if sha256(args.descriptor_dir / projection_name) != source['files_sha256'][projection_name]:
            raise ValueError('共同投影 SHA256 不符。')
        projection = torch.load(args.descriptor_dir / projection_name, map_location='cpu', weights_only=True)
        if (tuple(projection.shape) != tuple(source['response_projection']['shape'])
                or projection.shape[0] != int(source['model_config']['hidden_size'])
                or not np.isfinite(array(projection)).all()):
            raise ValueError('共同投影维度或数值非法。')

    plans = {}
    for method in METHODS:
        seeds = args.seeds[:1] if method == 'traffic_top' else args.seeds
        for seed in seeds:
            plans[method, seed] = {
                'format_version': 1, 'complete': False, 'model': model, 'method': method,
                'seed': seed, 'source_descriptor_dir': str(args.descriptor_dir),
                'source_descriptor_report_sha256': sha256(source_path),
                'source_checkpoint': source['checkpoint'], 'num_old_experts': old_n,
                'expand_to': args.expand_to, 'new_experts_per_layer': budget,
                'clusters': args.clusters if method not in ('random_copy', 'traffic_top') else None,
                'selection_data': 'fit only; never selected using held-out scores',
                'copy_rule': 'copy expert parameters and router row from the same parent; '
                             'no noise or parameter averaging; preserve shared experts and native top-k',
                'random_group_rule': 'permute full labels, preserving cluster sizes; '
                                     'recompute traffic budgets; use the same full-space representative rule',
                'layers': [],
            }
    metrics, normalizers, stability, null_samples = [], {}, [], []
    begin = time.perf_counter()
    log(f'模型={model}; {len(entries)} 层; {old_n}→{args.expand_to}; K={args.clusters}; seeds={args.seeds}')
    log('只读取描述符，CPU 分析；不加载模型权重。')
    for entry in entries:
        layer = int(entry['layer'])
        name = f'layer_{layer:02d}.pt'
        path = args.descriptor_dir / name
        if sha256(path) != source['files_sha256'][name]:
            raise ValueError(f'{name}: SHA256 不符。')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        validate_payload(payload, entry, source, {'fit': fit_indices, 'check': check_indices})
        blocks, normalizers[str(layer)] = normalized_blocks(payload['fit'])
        grams = {v: sum(blocks[b] @ blocks[b].T for b in names) for v, names in VARIANTS.items()}
        counts = array(payload['fit']['counts'])
        # The held-out response matrix never enters clustering, allocation or selection.
        check_distances = response_distances(payload['check']['responses'])
        solutions, groups = {}, {}
        for variant in VARIANTS:
            for seed in args.seeds:
                solution = fit_kmeans(grams[variant], args.clusters, seed + 1009 * layer,
                                      args.n_init, args.max_iter)
                solutions[variant, seed] = solution
                groups[variant, seed] = make_group_plan(grams[variant], counts, solution['labels'], budget)
            labels_list = [solutions[variant, seed]['labels'] for seed in args.seeds]
            aris = [adjusted_rand(a, b) for a, b in itertools.combinations(labels_list, 2)]
            stability.append({'layer': layer, 'variant': variant,
                              'seed_ari_mean': scalar_mean(aris),
                              'seed_ari_min': min(aris) if aris else None})
        for seed in args.seeds:
            full_labels = solutions['full', seed]['labels']
            random_labels = canonical_labels(np.random.default_rng(seed + 200003 + layer).permutation(full_labels))
            groups['random_groups', seed] = make_group_plan(grams['full'], counts, random_labels, budget)
            groups['random_copy', seed] = simple_plan(counts, budget, 'random_copy', seed + 300007 + layer)
        groups['traffic_top', args.seeds[0]] = simple_plan(counts, budget, 'traffic_top', args.seeds[0])

        for (method, seed), candidate in groups.items():
            parents = candidate['parents']
            if len(parents) != budget or min(parents) < 0 or max(parents) >= old_n:
                raise RuntimeError('复制父专家的数量或范围不合法。')
            plan_layer = {'layer': layer, 'num_old_experts': old_n, 'num_new_experts': budget,
                          'top_k': int(entry['top_k']), 'norm_topk_prob': payload['norm_topk_prob'],
                          'has_shared_expert': payload['has_shared_expert'],
                          'descriptor_sha256': source['files_sha256'][name], **candidate}
            plans[method, seed]['layers'].append(plan_layer)
            labels = np.asarray(candidate['labels']) if candidate['labels'] is not None else None
            ratio = response_ratio(check_distances, labels) if labels is not None else None
            reference = {'mean': None, 'q05': None, 'q95': None}
            if method in VARIANTS:
                reference = random_reference(check_distances, labels, args.random_groups,
                                             seed + 400009 + layer)
            metric = {
                'layer': layer, 'method': method, 'seed': seed,
                'check_within_over_all': ratio,
                'random_group_ratio_mean': reference['mean'],
                'random_group_ratio_q05': reference['q05'], 'random_group_ratio_q95': reference['q95'],
                'check_parent_coverage_ratio': parent_response_coverage(check_distances, parents, counts),
                'new_experts': budget, 'unique_parents': len(set(parents)),
                'traffic_vs_size_reallocation': candidate['traffic_vs_size_reallocation'],
                'ari_vs_full': adjusted_rand(labels, solutions['full', seed]['labels']) if labels is not None else None,
                'fit_inertia': solutions[method, seed]['inertia'] if method in VARIANTS else None,
            }
            metrics.append(metric)
        first = args.seeds[0]
        full_metric = next(x for x in metrics if x['layer'] == layer and x['method'] == 'full' and x['seed'] == first)
        ari = next(x['seed_ari_mean'] for x in stability if x['layer'] == layer and x['variant'] == 'full')
        ratio = full_metric['check_within_over_all']
        log(f'层 {layer:02d}: full seed-ARI={ari:.3f}; '
            f'留出簇内/全体距离={ratio:.3f}; 父专家种类={full_metric["unique_parents"]}/{budget}'
            if ratio is not None else f'层 {layer:02d}: 留出响应无有效距离，指标记为空值。')
        del payload, blocks, grams, check_distances

    # Preserve every candidate. Only K-means multiple starts use fit inertia.
    plan_dir = output / 'plans'
    plan_dir.mkdir()
    plan_files = {}
    for (method, seed), plan in plans.items():
        plan['layers'].sort(key=lambda row: row['layer'])
        plan['complete'] = True
        filename = f'{method}_seed{seed}.json'
        save_json(plan_dir / filename, plan)
        plan_files[filename] = sha256(plan_dir / filename)
    write_csv(output / 'layer_metrics.csv', metrics)
    write_csv(output / 'stability.csv', stability)
    save_json(output / 'normalization.json', normalizers)
    summary_rows = []
    for method in METHODS:
        selected = [x for x in metrics if x['method'] == method]
        variant_stability = [x for x in stability if x['variant'] == method]
        seed_means = [scalar_mean([x['check_within_over_all'] for x in selected if x['seed'] == seed])
                      for seed in sorted({x['seed'] for x in selected})]
        valid_means = [x for x in seed_means if x is not None]
        summary_rows.append({
            'method': method, 'layers': len(entries),
            'plan_seeds': len({x['seed'] for x in selected}),
            'seed_ari_mean': scalar_mean([x['seed_ari_mean'] for x in variant_stability]),
            'check_within_over_all_mean': scalar_mean([x['check_within_over_all'] for x in selected]),
            'check_ratio_across_seed_sd': float(np.std(valid_means, ddof=1)) if len(valid_means) > 1 else None,
            'random_group_ratio_mean': scalar_mean([x['random_group_ratio_mean'] for x in selected]),
            'check_parent_coverage_mean': scalar_mean([x['check_parent_coverage_ratio'] for x in selected]),
            'mean_unique_parents': scalar_mean([x['unique_parents'] for x in selected]),
        })
    write_csv(output / 'summary.csv', summary_rows)
    report = {
        'complete': True, 'model': model,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_report_sha256': sha256(source_path), 'plan_files_sha256': plan_files,
        'summary': summary_rows, 'elapsed_seconds': time.perf_counter() - begin,
        'numpy_version': np.__version__, 'torch_version': str(torch.__version__), 'python_version': sys.version,
        'script_sha256': sha256(__file__),
        'definitions': {
            'normalization': 'Each FIT block is centered across experts and divided by sqrt(mean squared norm). '
                             'Zero if centered energy <= max(1e-20, 1e-12 * raw energy). No PCA or spherical normalization.',
            'kmeans': 'Euclidean Lloyd + D^2 initialization; exact linear Gram algebra; best FIT inertia over n_init; '
                      'all outer seeds retained. This implementation is not sklearn bit-for-bit.',
            'seed_ari': 'Pairwise ARI across optimization seeds on the SAME fit data; not resampling stability.',
            'check_within_over_all': 'Mean squared response distance over same-cluster expert pairs divided by mean '
                                     'over ALL distinct expert pairs on CHECK probes. Unweighted expert pairs. '
                                     'Fixed-size random label permutations have expectation 1 when distances are nonzero. '
                                     'This denominator differs from older within/between scripts.',
            'random_quantiles': '5th/95th percentiles across size-matched random label permutations; '
                                'reference spread, not a data confidence interval or evidence of semantic specialization.',
            'parent_coverage': 'FIT-traffic-weighted distance on CHECK responses to the nearest selected parent, '
                               'divided by ALL-pair mean CHECK distance. Descriptive coverage, not mixed-output NMSE/PPL.',
            'summary': 'Equal layer weights; equal seed weights where available. Across-seed SD measures initialization '
                       'variation, not confidence across documents or training runs.',
            'random_copy': 'Uniform random permutation of old experts; no repeated parent until all have been used.',
            'traffic_top': 'Descending fit routed-token counts, then expert ID; cycle after all old experts are used. '
                           'Deterministic selection is exported once, not counted as independent random replicates.',
            'qwen': 'Expand only routed experts. Shared expert and its gate remain unchanged.',
        },
        'interpretation': 'Plans and diagnostics only. No expansion/LM training or performance improvement demonstrated. '
                          'Do not call these controls reproductions of published growth methods.',
    }
    save_json(output / 'report.json', report)
    for row in summary_rows:
        log(f'{row["method"]}: 留出比={row["check_within_over_all_mean"]}; '
            f'seed-ARI={row["seed_ari_mean"]}; 父专家覆盖距离={row["check_parent_coverage_mean"]}')
    log(f'完成，CPU 分析耗时 {report["elapsed_seconds"]:.1f} 秒；计划数={len(plans)}')
    log('结果目录: ' + str(output))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--descriptor-dir', type=Path, required=True)
    parser.add_argument('--expand-to', type=int, required=True)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--n-init', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=300)
    parser.add_argument('--random-groups', type=int, default=100)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('至少给出两个不同的非负随机种子。')
    if min(args.expand_to, args.n_init, args.max_iter, args.random_groups) < 1 or args.clusters < 2:
        parser.error('参数必须为正，K 至少为 2。')
    args.descriptor_dir = args.descriptor_dir.expanduser().resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main():
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = args.output_dir or args.descriptor_dir / f'groups_k{args.clusters}_to{args.expand_to}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args, output)
    except Exception as exc:
        save_json(output / 'failure.json', {'complete': False, 'error': str(exc),
                                         'error_type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
