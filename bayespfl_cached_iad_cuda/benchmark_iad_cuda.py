#!/usr/bin/env python3
"""Benchmark Bayes-PFL original vs Cached-IAD using torch.cuda.Event.

Run from the Bayes-PFL repository root, after applying the VPB.py patch.
Dataset loading / preprocessing happen BEFORE timings.
The measured model path includes visual encoding, prompt generation and RCA.
CUDA Events report elapsed time between events on the CUDA stream, including
potential stream idle time between kernel launches; they do not report pure GPU
kernel utilization or end-to-end HTTP request latency.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from models.VPB import Context_Prompting, TextEncoder
from models.model_CLIP import Load_CLIP, tokenize
from datasets import Makedataset
from test import _transform_test


class CudaTimer:
    """CUDA Event timing for nested blocks and module hooks, without synchronize in each block."""
    def __init__(self):
        self.enabled = False
        self.events = defaultdict(list)
        self.in_flight = defaultdict(list)
        self.text_index = 0
        self.text_stack = []

    def reset(self):
        self.events.clear()
        self.in_flight.clear()
        self.text_index = 0
        self.text_stack.clear()

    def start(self, name):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.in_flight[name].append(ev)

    def stop(self, name):
        if not self.enabled:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        begin = self.in_flight[name].pop()
        self.events[name].append((begin, end))

    def get_ms(self):
        # Requires torch.cuda.synchronize before calling.
        if any(v for v in self.in_flight.values()):
            raise RuntimeError('Unbalanced timer.start()/stop()')
        return {k: sum(s.elapsed_time(e) for s, e in v) for k, v in self.events.items()}


def register_module_timing(model, text_encoder, timer):
    handles = []
    for module, name in (
        (model.PFL_context, 'ISD_context'),
        (model.PFL_normal, 'IAD_normal'),
        (model.PFL_abnormal, 'IAD_abnormal'),
    ):
        handles.append(module.register_forward_pre_hook(
            lambda _module, _args, key=name: timer.start(key)
        ))
        handles.append(module.register_forward_hook(
            lambda _module, _args, _out, key=name: timer.stop(key)
        ))

    def text_pre(_module, _args):
        name = 'TextEncoder_normal' if timer.text_index % 2 == 0 else 'TextEncoder_abnormal'
        timer.text_index += 1
        timer.text_stack.append(name)
        timer.start(name)

    def text_post(_module, _args, _out):
        timer.stop(timer.text_stack.pop())

    handles.append(text_encoder.register_forward_pre_hook(text_pre))
    handles.append(text_encoder.register_forward_hook(text_post))
    return handles


@torch.inference_mode()
def infer(model_clip, model, text_encoder, image, cls_name, device, args, stage, timer):
    timer.start('model_total')

    timer.start('CLIP_Image_Encoder')
    image_features, _, patch_tokens = model_clip.encode_image(image, args.features_list)
    timer.stop('CLIP_Image_Encoder')

    timer.start('Prompt_Generation_Total')
    text_embeddings, _ = model.forward_ensemble(
        text_encoder, image_features, patch_tokens,
        cls_name, device, tokenize, mode='test'
    )
    timer.stop('Prompt_Generation_Total')

    timer.start('Localization_Total')
    pro_img, anomaly_maps_list = model(
        text_embeddings, image_features, patch_tokens,
        stage=stage, mode='test'
    )
    timer.stop('Localization_Total')

    timer.stop('model_total')
    return pro_img, anomaly_maps_list


def run_variant(variant, model_clip, model, text_encoder, samples, args, stage, device, timer):
    cache_enabled = (variant == 'cached_iad')
    model.set_iad_cache(cache_enabled)
    model.eval()
    timer.enabled = False

    # Warm-up in BOTH variants: identical seeds, identical image sequence, and
    # the first IAD samples are cached. Cache-miss cost is therefore excluded
    # from steady-state inference latencies, but measured separately below.
    for i in range(args.warmup):
        torch.manual_seed(args.seed + i)
        image, cls_name = samples[i % len(samples)]
        infer(model_clip, model, text_encoder, image, cls_name, device, args, stage, timer)
    torch.cuda.synchronize(device)

    rows = []
    predictions = []
    keys = [
        'model_total', 'CLIP_Image_Encoder', 'Prompt_Generation_Total',
        'ISD_context', 'IAD_normal', 'IAD_abnormal',
        'TextEncoder_normal', 'TextEncoder_abnormal', 'Localization_Total'
    ]

    for i in range(args.iterations):
        image, cls_name = samples[i % len(samples)]
        # Resetting the RNG immediately before each inference isolates the
        # effect of skipped IAD random draws on subsequent ISD samples.
        # Important: this is an A/B verification protocol, not the exact
        # stochastic sequence produced by unmodified test.py.
        torch.manual_seed(args.seed + 10_000 + i)
        timer.reset()
        timer.enabled = True
        logits, maps = infer(
            model_clip, model, text_encoder, image, cls_name,
            device, args, stage, timer
        )
        timer.enabled = False
        torch.cuda.synchronize(device)
        observed = timer.get_ms()
        row = {'variant': variant, 'step': i + 1}
        row.update({k: float(observed.get(k, 0.0)) for k in keys})
        rows.append(row)

        if i < args.verify_images:
            predictions.append((
                logits.detach().float().cpu().clone(),
                [m.detach().float().cpu().clone() for m in maps]
            ))
    return rows, predictions


def compare_outputs(a, b):
    max_abs = 0.0
    n = 0
    for (loga, mapa), (logb, mapb) in zip(a, b):
        for x, y in ((loga, logb), *zip(mapa, mapb)):
            if x.shape != y.shape:
                raise RuntimeError(f'Model output shape mismatch: {x.shape} vs {y.shape}')
            max_abs = max(max_abs, float((x - y).abs().max()))
            n += 1
    return max_abs, n


def main():
    parser = argparse.ArgumentParser('Bayes-PFL Cached-IAD CUDA Events benchmark')
    parser.add_argument('--dataset', default='visa')
    parser.add_argument('--data_path', default='./dataset/mvisa/data')
    parser.add_argument('--image_size', type=int, default=518)
    parser.add_argument('--features_list', nargs='+', type=int, default=[6, 12, 18, 24])
    parser.add_argument('--config_path', default='./open_clip_local/model_configs/ViT-L-14-336.json')
    parser.add_argument('--pretrained_path', default='./pretrained_weight/ViT-L-14-336px.pt')
    parser.add_argument('--checkpoint_path', default='./bayes_weight/train_visa.pth')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--prompt_context_len', type=int, default=5)
    parser.add_argument('--prompt_state_len', type=int, default=5)
    parser.add_argument('--prompt_num', type=int, default=3)
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--num_flows', type=int, default=10)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--num_images', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--verify_images', type=int, default=5)
    parser.add_argument('--output_csv', default='iad_cuda_benchmark.csv')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU is required for torch.cuda.Event timings.')
    if args.warmup < 1 or args.iterations < 1 or args.num_images < 1:
        raise SystemExit('warmup, iterations, num_images must be >= 1.')
    if not hasattr(Context_Prompting, 'set_iad_cache'):
        raise SystemExit('models/VPB.py has not been patched. Apply apply_variant_a.py first.')

    torch.cuda.set_device(args.device_id)
    device = torch.device(f'cuda:{args.device_id}')
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    args.vision_width = config['vision_cfg']['width']
    args.text_width = config['text_cfg']['width']
    args.embed_dim = config['embed_dim']

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_clip, _, _ = Load_CLIP(args.image_size, args.pretrained_path, device=device)
    model_clip = model_clip.to(device).eval()
    text_encoder = TextEncoder(model_clip, args).to(device).eval()
    model = Context_Prompting(args=args).to(device).eval()
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['MyModel'], strict=True)
    model.clear_iad_cache()

    preprocess_test = _transform_test(args.image_size)
    maker = Makedataset(
        train_data_path=args.data_path,
        preprocess_test=preprocess_test,
        mode='test',
        image_size=args.image_size,
    )
    loader, _ = maker.mask_dataset(
        name=args.dataset, product_list=None, batchsize=1, shuf=False
    )
    samples = []
    for batch in loader:
        samples.append((batch['img'].to(device), batch['cls_name']))
        if len(samples) >= args.num_images:
            break
    if not samples:
        raise SystemExit('Test dataset contains no images.')

    stage = int(args.checkpoint_path[:-4].split('_')[-1]) if 'epoch_post' in args.checkpoint_path else 2
    print(f'GPU: {torch.cuda.get_device_name(device)}')
    print(f'Images preloaded on GPU: {len(samples)}; warm-up: {args.warmup}; '
          f'iterations per variant: {args.iterations}; stage={stage}')

    timer = CudaTimer()
    handles = register_module_timing(model, text_encoder, timer)
    try:
        original_rows, original_predictions = run_variant(
            'original', model_clip, model, text_encoder, samples,
            args, stage, device, timer
        )
        cached_rows, cached_predictions = run_variant(
            'cached_iad', model_clip, model, text_encoder, samples,
            args, stage, device, timer
        )
    finally:
        for h in handles:
            h.remove()

    columns = list(original_rows[0].keys())
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(original_rows + cached_rows)
    print(f'\nSaved: {args.output_csv}')

    keys = [k for k in columns if k not in ('variant', 'step')]
    print('\nCUDA Event stream elapsed time, milliseconds (mean / median / p95):')
    for key in keys:
        a = np.array([r[key] for r in original_rows], dtype=float)
        b = np.array([r[key] for r in cached_rows], dtype=float)
        print(f'{key:25s}  original {a.mean():8.3f} / {np.median(a):8.3f} / {np.percentile(a,95):8.3f}'
              f'   cached {b.mean():8.3f} / {np.median(b):8.3f} / {np.percentile(b,95):8.3f}')

    a_total = float(np.median([r['model_total'] for r in original_rows]))
    b_total = float(np.median([r['model_total'] for r in cached_rows]))
    print(f'\nSteady-state speedup (median total model latency): {a_total / b_total:.3f}x')
    max_abs, tensors = compare_outputs(original_predictions, cached_predictions)
    print(f'Output max absolute difference across {tensors} tensors: {max_abs:.9g}')
    if max_abs > 1e-4:
        print('WARNING: outputs differ materially; investigate stochastic state / numerical precision.')
    else:
        print('Outputs agree within 1e-4 under per-inference RNG reseeding.')


if __name__ == '__main__':
    main()
