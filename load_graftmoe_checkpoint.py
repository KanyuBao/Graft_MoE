#!/usr/bin/env python3
"""Reload completed GRAFT-MoE runs with native inference routing.

Python API: model, tokenizer = load_run('/path/to/runs/olmoe_graftmoe_seed42')
Expanded runs require final_delta; full-parameter continuation requires
final_model. Original runs reload the recorded source checkpoint.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required file is missing: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _file_digest(value):
    result = value.get('sha256') if isinstance(value, dict) else value
    if not isinstance(result, str) or len(result) != 64 or any(c not in '0123456789abcdef' for c in result):
        raise ValueError('Checkpoint manifest contains an invalid SHA256 digest.')
    return result


def _check_file_list(directory, files):
    if not isinstance(files, dict) or not files:
        raise ValueError(f'Empty checkpoint file manifest: {directory}')
    for name, value in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f'Checkpoint file escapes its directory: {name}')
        _file_digest(value)
        if not path.is_file():
            raise FileNotFoundError(f'Retained checkpoint file is missing: {path}')


def inspect_run(run_dir):
    """Validate saved metadata and required files without loading model tensors.

    Actual source identity and retained-weight checksums are verified by load_run.
    The frozen recipe is used instead of the current editable project config.
    """
    run = Path(run_dir).expanduser().resolve()
    recipe = _read(run/'recipe.json')
    complete = _read(run/'complete.json')
    if recipe.get('format') != 'graftmoe-v1':
        raise ValueError('Expected a graftmoe-v1 recipe; use the matching package for older runs.')
    if not complete.get('complete') or complete.get('recipe_hash') != _digest(recipe):
        raise ValueError('The run is incomplete or complete.json does not match its frozen recipe.')
    for key in ('model', 'method', 'seed'):
        if complete.get(key) != recipe[key]:
            raise ValueError(f'Completed-run {key} differs from the saved recipe.')
    cfg, key, method = recipe['config'], recipe['model'], recipe['method']
    if method != 'original' and complete.get('final_step') != cfg['steps']:
        raise ValueError('The run has not reached its configured final training step.')
    source = Path(cfg['models'][key]['path']).expanduser()
    if not source.is_absolute():
        raise ValueError('The frozen recipe must contain a resolved absolute source-model path.')
    if not (source/'config.json').is_file():
        raise FileNotFoundError(f'Recorded source checkpoint is unavailable: {source}')
    result = {'run': run, 'recipe': recipe, 'complete': complete,
              'kind': 'original', 'manifest': None, 'plan': None, 'directory': source}
    if method == 'original':
        return result
    if method == 'continue':
        directory = run/'final_model'
        manifest_path = directory/'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f'Full-parameter continuation requires retained weights: {manifest_path}. '
                'Run with save_final_weights=true; scalar metrics cannot reconstruct model parameters.')
        manifest = _read(manifest_path)
        if manifest.get('format') != 'hf-pretrained' or manifest.get('method') != method:
            raise ValueError('Unexpected full-model checkpoint manifest.')
        _check_file_list(directory, manifest['files'])
        if 'config.json' not in manifest['files']:
            raise ValueError('The retained full-model manifest must include config.json.')
        result.update(kind='full_model', directory=directory, manifest=manifest)
        return result
    plan = _read(run/'plan.json')
    if plan != recipe.get('plan'):
        raise ValueError('plan.json differs from the expansion plan saved in recipe.json.')
    directory = run/'final_delta'
    manifest_path = directory/'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'Expanded-model weights were not retained: {manifest_path}. '
            'Run with save_final_weights=true; plan.json and metrics alone cannot restore learned weights.')
    manifest = _read(manifest_path)
    model_cfg = cfg['models'][key]
    old, total = model_cfg['old_experts'], model_cfg['expand_to']
    if manifest.get('method') != method or manifest.get('old_experts') != old:
        raise ValueError('Final-delta metadata differs from the training recipe.')
    if total <= old or plan.get('expand_to') != total:
        raise ValueError('Invalid expanded expert count in the saved plan.')
    layer_ids = [row['layer'] for row in plan['layers']]
    if not layer_ids or len(layer_ids) != len(set(layer_ids)):
        raise ValueError('Expansion plan must contain unique MoE layers.')
    expected_files = {f'layer_{layer:02d}.pt' for layer in layer_ids}
    if set(manifest['files']) != expected_files:
        raise ValueError('Final-delta file list does not match the expansion plan.')
    for row in plan['layers']:
        parents = row['parents']
        if row['num_old_experts'] != old or row['top_k'] != model_cfg['top_k']:
            raise ValueError('Expansion plan architecture differs from the recorded model.')
        if len(parents) != total-old or any(type(p) is not int or not 0 <= p < old for p in parents):
            raise ValueError('Invalid source-parent mapping in the expansion plan.')
    _check_file_list(directory, manifest['files'])
    result.update(kind='delta', directory=directory, manifest=manifest, plan=plan)
    return result


def load_run(run_dir, device='cuda:0'):
    """Return (model, tokenizer), frozen in eval mode with exploration disabled."""
    info = inspect_run(run_dir)
    import torch
    from transformers import AutoTokenizer
    import baseline_utils as bu
    import graftmoe_core as c

    recipe = info['recipe']
    cfg, key = recipe['config'], recipe['model']
    identity = bu.checkpoint_identity(Path(cfg['models'][key]['path']))
    for field in ('config_sha256', 'index_sha256', 'weight_files'):
        if identity[field] != recipe['base_checkpoint'][field]:
            raise ValueError(f'The source checkpoint differs from the recorded training source ({field}).')
    if info['manifest'] is not None:
        for name, digest in info['manifest']['files'].items():
            path = info['directory']/name
            if bu.sha256(path) != _file_digest(digest):
                raise ValueError(f'Retained-weight checksum failed: {path}')
    model_cfg = copy.deepcopy(cfg)
    if info['kind'] == 'full_model':
        model_cfg['models'][key]['path'] = str(info['directory'])
    model = c.load_model(model_cfg, key, device)
    if info['kind'] == 'delta':
        c.apply_expansion(model, info['plan'], .001 if recipe['method']=='eu_gn' else 0., recipe['seed'])
        # Native modules have exactly the saved gate/expert parameter names;
        # training-only exploration wrappers are unnecessary for inference.
        with torch.no_grad():
            for idx, moe, _, _ in bu.iter_moe_blocks(model):
                path = info['directory']/f'layer_{idx:02d}.pt'
                saved = c.load_pt(path)
                parameters = dict(moe.named_parameters())
                expected = {'gate.'+name for name, _ in moe.gate.named_parameters()}
                for expert in range(info['manifest']['old_experts'], len(moe.experts)):
                    expected |= {f'experts.{expert}.'+name for name, _ in moe.experts[expert].named_parameters()}
                if set(saved) != expected:
                    raise ValueError(f'Final-delta parameter names differ from the rebuilt architecture: {path}')
                for name, value in saved.items():
                    if not isinstance(value, torch.Tensor) or value.shape != parameters[name].shape:
                        raise ValueError(f'Invalid shape for saved parameter {name}: {path}')
                    if not torch.isfinite(value).all():
                        raise ValueError(f'Non-finite saved parameter {name}: {path}')
                    parameters[name].copy_(value)
    for _, moe, _, _ in bu.iter_moe_blocks(model):
        if hasattr(moe, 'exploration'):
            moe.exploration = 0.
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['models'][key]['path'], local_files_only=True)
    return model.eval().requires_grad_(False), tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True, help='Completed run containing recipe.json and complete.json.')
    parser.add_argument('--device', default='cuda:0', help='PyTorch device; CUDA index is after CUDA_VISIBLE_DEVICES mapping.')
    parser.add_argument('--check', action='store_true', help='Check metadata and file presence only; no tensor loading or GPU use.')
    args = parser.parse_args()
    info = inspect_run(args.run_dir)
    if args.check:
        print(f"Checkpoint metadata ready: {info['run']} ({info['kind']}). Weight checksums are checked on load.")
        return
    model, _ = load_run(args.run_dir, args.device)
    print(f"Loaded {info['recipe']['model']}/{info['recipe']['method']} on {args.device}; "
          f"{sum(p.numel() for p in model.parameters()):,} parameters; native routing, eval mode, gradients disabled.")


if __name__ == '__main__':
    main()
