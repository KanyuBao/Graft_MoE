#!/usr/bin/env python3
"""Aggregate measured results only. Incomplete matrices are explicitly marked.
No smoothing, seed selection, or synthetic performance values.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

LABELS = {'original': 'Original checkpoint', 'continue': 'Continue original (all parameters)',
          'random_copy': 'Random copy + LB', 'traffic_copy': 'Traffic copy + LB',
          'eu_gn': 'EU-GN (adapted)', 'graftmoe': 'GRAFT-MoE', 'graftmoe_native': 'w/o routing exploration',
          'cluster_no_warm': 'GRAFT-MoE w/o warm-start', 'random_groups_warm': 'Random groups + warm-start'}
COLORS = {'original': '#666666', 'continue': '#333333', 'random_copy': '#4C78A8',
          'traffic_copy': '#B279A2', 'eu_gn': '#54A24B', 'graftmoe': '#E45756', 'graftmoe_native': '#9D755D',
          'cluster_no_warm': '#F2A541', 'random_groups_warm': '#72B7B2'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean_sd(xs):
    values = [float(v) for v in xs if v is not None and math.isfinite(float(v))]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values)>1 else None


def paired_interval(deltas, seed=924, reps=2000, block_group=32):
    """Hierarchical paired seed + contiguous packed-block bootstrap.
    Resample training seeds and 32-block groups; same resampling for both methods.
    Interval is descriptive with only 3 training seeds; not a corrected p-value.
    """
    a = np.asarray(deltas, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] % block_group:
        raise ValueError('Need [paired seeds, blocks] with blocks divisible by group size.')
    groups = a.reshape(a.shape[0], -1, block_group).mean(-1)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for i in range(reps):
        seeds = rng.integers(len(groups), size=len(groups))
        blocks = rng.integers(groups.shape[1], size=groups.shape[1])
        draws[i] = groups[np.ix_(seeds, blocks)].mean()
    return float(a.mean()), float(np.quantile(draws,.025)), float(np.quantile(draws,.975))


def curve(run):
    rows = [read(p) for p in sorted(run.glob('valid_step*.json'))]
    # Eval at each optimizer step is overwritten deterministically after resume.
    return {r['step']: r for r in rows}


def main(root):
    root = Path(root).expanduser().resolve()
    cfg = read(root/'protocol_frozen.json')
    requested = read(root/'requested_matrix.json')
    methods = list(dict.fromkeys(requested['methods']))
    expected = {(m,'original',0) for m in requested['models']}
    expected |= {(m,method,s) for m in requested['models'] for method in methods for s in cfg['seeds']}
    out = root/'summary'
    out.mkdir(exist_ok=True)
    records = []
    keyed = {}
    excluded = []
    for p in sorted((root/'runs').glob('*/complete.json')):
        report = read(p)
        key = (report['model'],report['method'],report['seed'])
        if key not in expected:
            excluded.append({'path': str(p), 'reason': 'outside requested matrix'})
            continue
        if not report.get('complete') or (report['method'] != 'original' and report.get('final_step') != cfg['steps']):
            excluded.append({'path': str(p), 'reason': 'configured training run is incomplete'})
            continue
        if key in keyed:
            raise ValueError(f'Duplicate completed result for {key}: {p}')
        report['_path'] = str(p.parent)
        keyed[key] = report
        records.append(report)
    missing = sorted(expected-set(keyed))
    (out/'completeness.json').write_text(json.dumps({'complete': not missing,
       'expected_runs': len(expected), 'finished_runs': len(expected & set(keyed)), 'missing': missing,
       'excluded': excluded}, indent=2), encoding='utf-8')
    rows, run_rows, recovery_rows, curve_rows, differences = [], [], [], [], []
    for r in records:
        path = Path(r['_path'])
        benchmark = r.get('benchmark') or {}
        row = {'model': r['model'], 'method': r['method'], 'seed': r['seed'], 'input_tokens': r['input_tokens'],
               'test_nll': r['test']['nll'], 'test_ppl': r['test']['ppl'],
               'test_cv': r['test']['mean_load_cv'], 'test_new_share': r['test']['new_assignment_share'],
               'valid_nll': r['validation']['nll'], 'parameters': r['num_parameters'],
               'trainable_parameters': r['trainable_parameters'], 'lm_seconds': r['train_seconds'],
               'warm_seconds': r['warm_seconds'], 'prefill_tokens_per_s': benchmark.get('input_tokens_per_second'),
               'decode_ms_per_token': benchmark.get('decode_milliseconds_per_token'),
               'eval_peak_gib': r['validation']['peak_allocated_gib']}
        logs = [json.loads(line) for line in (path/'train.jsonl').read_text().splitlines()] if (path/'train.jsonl').exists() else []
        row['train_peak_gib'] = max((x['peak_allocated_gib'] for x in logs),default=None)
        for task in cfg['tasks']:
            value = r['tasks'].get(task, {}).get('score')
            row[task] = 100*value if value is not None else None
        task_scores = [row[task] for task in cfg['tasks'] if row[task] is not None and math.isfinite(row[task])]
        row['tasks_measured'] = len(task_scores)
        row['tasks_expected'] = len(cfg['tasks'])
        row['task_average'] = float(np.mean(task_scores)) if task_scores and len(task_scores)==len(cfg['tasks']) else None
        run_rows.append(row)
        if r['method'] != 'original':
            c = curve(path)
            source = read(path/'source_quick.json')
            reference = source['nll']
            if any(v.get('indices_sha256') != source.get('indices_sha256') for v in c.values()):
                raise ValueError(f'Recovery evaluations use different validation indices: {path}')
            steps = sorted(c)
            tokens_per_step = r['input_tokens'] / r['final_step']
            if tokens_per_step <= 0 or not tokens_per_step.is_integer():
                raise ValueError(f'Invalid recorded input-token budget: {path}')
            tokens = np.array(steps)*int(tokens_per_step)
            nll = np.array([c[s]['nll'] for s in steps])
            recovered = [int(t) for t, v in zip(tokens,nll) if v <= reference]
            trapezoid = getattr(np, 'trapezoid', None)
            if trapezoid is None:
                trapezoid = np.trapz
            auc = float(trapezoid(np.maximum(0., nll-reference), tokens)) if len(tokens)>1 else None
            recovery_rows.append({'model': r['model'], 'method': r['method'], 'seed': r['seed'],
                                  'source_quick_nll': reference,
                                  'first_observed_recovery_input_tokens': recovered[0] if recovered else None,
                                  'right_censored': not recovered if steps else None,
                                  'observed_through_input_tokens': int(tokens[-1]) if len(tokens) else 0,
                                  'positive_excess_nll_token_auc': auc,
                                  'warm_cost_excluded_from_token_axis_seconds': r['warm_seconds']})
            for s in steps:
                curve_rows.append({'model': r['model'], 'method': r['method'], 'seed': r['seed'],
                                   'step': s, 'input_tokens': s*int(tokens_per_step),
                                   'quick_valid_nll': c[s]['nll'], 'quick_valid_ppl': c[s]['ppl'],
                                   'new_assignment_share': c[s]['new_assignment_share'],
                                   'lm_seconds': c[s]['train_seconds']})
    for model in requested['models']:
        for method in ['original']+methods:
            selected = [r for r in run_rows if r['model']==model and r['method']==method]
            row = {'model': model, 'method': method, 'label': LABELS[method], 'seeds_finished': len(selected),
                   'seeds_expected': 1 if method=='original' else len(cfg['seeds'])}
            for key in ('test_nll','test_ppl','test_cv','test_new_share','valid_nll','lm_seconds',
                        'warm_seconds','prefill_tokens_per_s','decode_ms_per_token','train_peak_gib',
                        'eval_peak_gib','task_average',*cfg['tasks']):
                row[key+'_mean'],row[key+'_sd'] = mean_sd([r.get(key) for r in selected])
            rows.append(row)
        for baseline in [m for m in methods if m!='graftmoe']:
            pairs = []
            seeds = []
            for seed in cfg['seeds']:
                ours = keyed.get((model,'graftmoe',seed))
                other = keyed.get((model,baseline,seed))
                if not ours or not other:
                    continue
                a,b = Path(ours['_path']),Path(other['_path'])
                ia = np.load(a/'test_final_indices.npy', allow_pickle=False)
                ib = np.load(b/'test_final_indices.npy', allow_pickle=False)
                if not np.array_equal(ia,ib):
                    raise ValueError('Paired test indices differ.')
                pairs.append(np.load(a/'test_final_block_nll.npy',allow_pickle=False)-np.load(b/'test_final_block_nll.npy',allow_pickle=False))
                seeds.append(seed)
            if pairs:
                mean = float(np.mean(pairs))
                lo,hi = None,None
                interval_note = 'paired seed+32-block-group bootstrap; descriptive, no multiple-comparison correction'
                if len(pairs[0]) and len(pairs[0]) % 32 == 0:
                    mean,lo,hi = paired_interval(pairs)
                else:
                    interval_note = 'interval not calculated: packed-block count is not divisible by 32'
                differences.append({'model': model, 'contrast': f'graftmoe - {baseline}', 'paired_seeds': len(pairs),
                                    'delta_nll': mean, 'bootstrap_95_low': lo, 'bootstrap_95_high': hi,
                                    'ppl_ratio_geomean': math.exp(mean),
                                    'direction': 'negative delta NLL favors GRAFT-MoE',
                                    'note': interval_note})
    for name, data in [('main_results.csv', rows),('per_run_results.csv',run_rows),('paired_differences.csv',differences),
                       ('recovery.csv',recovery_rows),('curves.csv',curve_rows)]:
        write_csv(out/name,data)
    # Keep all data even when incomplete; flag both TeX and figures clearly.
    lines = ['% Generated from complete.json; mean +/- sample SD across training seeds.',
             '% INCOMPLETE REQUESTED MATRIX: see completeness.json and per-method seed counts.' if missing else '% Requested matrix complete.',
             r'\begin{tabular}{llrrr}',r'\toprule',
             r'Model & Method & Test PPL $\downarrow$ & Test NLL $\downarrow$ & Task Avg. $\uparrow$ \\',r'\midrule']
    for row in rows:
        def fmt(key, digits=3):
            mean,sd = row[key+'_mean'],row[key+'_sd']
            if mean is None:
                return '--'
            return f'{mean:.{digits}f}' if sd is None else f'${mean:.{digits}f} \\pm {sd:.{digits}f}$'
        label = row['label'].replace('_',r'\_')
        lines.append(f"{row['model']} & {label} & {fmt('test_ppl')} & {fmt('test_nll')} & {fmt('task_average',2)} " + r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (out/'main_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if curve_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                             'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
        for xfield, xlabel, filename in [('input_tokens','CPT input tokens (millions)','recovery_tokens'),
                                        ('lm_seconds','CPT update wall time (hours)','recovery_time')]:
            fig, axes = plt.subplots(1,len(requested['models']),figsize=(6.2*len(requested['models']),4.0),squeeze=False)
            for model, ax in zip(requested['models'],axes[0]):
                original = keyed.get((model,'original',0))
                if original:
                    ref = read(Path(original['_path'])/'source_quick.json')['nll']
                    ax.axhline(ref,color='#666666',linestyle='--',linewidth=1.4,label='Source checkpoint')
                for method in methods:
                    data = [r for r in curve_rows if r['model']==model and r['method']==method]
                    common = sorted(set(r['step'] for r in data))
                    x,y,sd = [],[],[]
                    for s in common:
                        subset = [r for r in data if r['step']==s]
                        x.append(float(np.mean([r[xfield] for r in subset]))/(1e6 if xfield=='input_tokens' else 3600))
                        values = [r['quick_valid_nll'] for r in subset]
                        y.append(float(np.mean(values)))
                        sd.append(float(np.std(values,ddof=1)) if len(values)>1 else np.nan)
                    if not x:
                        continue
                    ax.plot(x,y,color=COLORS[method],linewidth=2.2 if method=='graftmoe' else 1.6,
                            marker='o',markersize=3,label=LABELS[method])
                    ax.fill_between(x,np.array(y)-sd,np.array(y)+sd,color=COLORS[method],alpha=.12,linewidth=0)
                ax.set(title=model.upper(),xlabel=xlabel,ylabel=f"Validation NLL (fixed {cfg['quick_valid_blocks']}-block subset)")
                ax.grid(alpha=.18)
            handles,labels = axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=3,frameon=False)
            if missing:
                fig.suptitle('INCOMPLETE REQUESTED MATRIX',color='#B91C1C')
            fig.tight_layout()
            for ext in ('pdf','png'):
                fig.savefig(out/(filename+'.'+ext),dpi=240)
            plt.close(fig)
    (out/'INTERPRETATION.txt').write_text(
        'PPL is compared within each model/tokenizer only. Final test uses the fixed last step.\n'
        'Means and sample SD use completed requested seeds; see per-method seed counts and completeness.json.\n'
        'One seed has no sample SD. Original has one deterministic checkpoint.\n'
        'Task average is reported only when every requested downstream task has a measured score.\n'
        'Recovery uses the SAME fixed random validation subset for source and adapted models; no full/subset mixing.\n'
        'Recovery is first observed grid crossing, not exact recovery time; no crossing is right-censored.\n'
        'CPT wall time includes CPU optimizer update, excludes calibration/warm/evaluation/checkpoint I/O.\n'
        'Time-axis bands reflect equal optimizer steps; x is mean recorded update wall time.\n'
        'PPL/benchmark improvement supports performance on the measured distributions. Activation alone does not establish specialization.\n'
        'Paired intervals are descriptive; three seeds do not guarantee stable significance or all-domain generalization.\n'
        'No from-scratch larger-MoE comparison was run by this package.\n',encoding='utf-8')
    print(f'Summary: {out}; complete={not missing}; finished={len(keyed)}/{len(expected)}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
