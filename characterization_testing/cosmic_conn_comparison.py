"""Compare cosmic-conn's ground_imaging and NRES models to astroscrappy on injected frames.
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cosmic_conn import init_model

from astroscrappy_study import (download_frame, load_frame, score_detection, CR_FRAME_PAIRS,
                                harvest_cr_stamps, inject_stamps, score_injected_frame)
from banzai_floyds.tests.utils import generate_fake_science_frame
from banzai_floyds.cosmics import CosmicRayDetector

OUTPUT_PDF = 'cosmic_conn_comparison.pdf'
COSMIC_CONN_CROP = 256  # matches banzai.cosmic.CosmicRayDetector's override of the model default
COSMIC_CONN_MODELS = ['ground_imaging', 'NRES']
THRESHOLD_GRID = np.concatenate([[0.001, 0.005], np.arange(0.01, 1.0, 0.02)])
SIGCLIP_GRID = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

COSMIC_CONN_COLORS = {'ground_imaging': '#D55E00', 'NRES': '#009E73'}
ASTROSCRAPPY_COLOR = '#0072B2'


def run_cosmic_conn(model_name: str, stamps: list, seeds: list, n_injections: int,
                    on_trace_fraction: float) -> list:
    """Inject one synthetic frame per seed, run cosmic-conn once per frame, and score every
    threshold in THRESHOLD_GRID against the same truth used for astroscrappy.
    """
    cr_model = init_model(model_name)
    cr_model.opt.crop = COSMIC_CONN_CROP

    results = []
    for seed in seeds:
        np.random.seed(seed)  # generate_fake_science_frame draws sky lines with the legacy np.random API
        frame = generate_fake_science_frame(include_sky=True, flat_spectrum=False)
        rng = np.random.default_rng(seed)
        truth, _ = inject_stamps(frame, stamps, n_injections, rng, on_trace_fraction)
        valid = frame.orders.data > 0

        cr_prob = cr_model.detect_cr(frame.data.astype(np.float32))
        for threshold in THRESHOLD_GRID:
            counts = score_detection(cr_prob > threshold, truth, valid)
            results.append({'seed': seed, 'threshold': threshold, **counts})
        print(f'cosmic-conn ({model_name}) seed={seed} done')
    return results


def run_astroscrappy(stamps: list, seeds: list, n_injections: int, on_trace_fraction: float) -> list:
    """Score astroscrappy at the adopted objlim/sigfrac, sweeping sigclip, on the same frames."""
    results = []
    for seed in seeds:
        np.random.seed(seed)
        frame = generate_fake_science_frame(include_sky=True, flat_spectrum=False)
        rng = np.random.default_rng(seed)
        truth, read_noise = inject_stamps(frame, stamps, n_injections, rng, on_trace_fraction)

        for sigclip in SIGCLIP_GRID:
            counts, _ = score_injected_frame(frame, truth, read_noise, sigclip, CosmicRayDetector.SIGFRAC,
                                             CosmicRayDetector.OBJLIM, use_background=True)
            results.append({'seed': seed, 'sigclip': sigclip, **counts})
        print(f'astroscrappy seed={seed} done')
    return results


def aggregate(results: list, key: str) -> list:
    """Sum pixel counts over seeds for each value of `key` and derive completeness/fdr."""
    totals = {}
    for row in results:
        entry = totals.setdefault(row[key], {'tp': 0, 'fp': 0, 'fn': 0, 'n_clean': 0})
        for count_key in entry:
            entry[count_key] += row[count_key]
    aggregated = []
    for value, entry in sorted(totals.items()):
        n_true = entry['tp'] + entry['fn']
        n_detected = entry['tp'] + entry['fp']
        aggregated.append({
            key: value, **entry,
            'completeness': entry['tp'] / n_true if n_true else np.nan,
            'false_discovery_rate': entry['fp'] / n_detected if n_detected else np.nan,
        })
    return aggregated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-injections', type=int, default=300)
    parser.add_argument('--on-trace-fraction', type=float, default=0.5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()

    stamps = []
    for basename1, basename2 in CR_FRAME_PAIRS:
        frame1 = load_frame(download_frame(basename1))
        frame2 = load_frame(download_frame(basename2))
        stamps += harvest_cr_stamps(frame1, frame2)
    print(f'Harvested {len(stamps)} real cosmic-ray stamps from the quiet, off-order region')

    cc_aggregated = {}
    for model_name in COSMIC_CONN_MODELS:
        cc_results = run_cosmic_conn(model_name, stamps, args.seeds, args.n_injections, args.on_trace_fraction)
        cc_aggregated[model_name] = aggregate(cc_results, 'threshold')

    as_results = run_astroscrappy(stamps, args.seeds, args.n_injections, args.on_trace_fraction)
    as_aggregated = aggregate(as_results, 'sigclip')

    for model_name, aggregated in cc_aggregated.items():
        print(f'\ncosmic-conn ({model_name}, raw data, threshold swept):')
        for row in aggregated:
            print(f"  threshold={row['threshold']:.3f}  completeness={row['completeness']:.3f}  "
                  f"fdr={row['false_discovery_rate']:.3f}")
    print(f"\nastroscrappy (objlim={CosmicRayDetector.OBJLIM:g}, sigfrac={CosmicRayDetector.SIGFRAC:g}, "
          f"inbkg on, sigclip swept):")
    for row in as_aggregated:
        print(f"  sigclip={row['sigclip']:g}  completeness={row['completeness']:.3f}  "
              f"fdr={row['false_discovery_rate']:.3f}")

    with PdfPages(OUTPUT_PDF) as pdf:
        fig, ax = plt.subplots(figsize=(7, 6))
        for model_name, aggregated in cc_aggregated.items():
            ax.plot([r['false_discovery_rate'] for r in aggregated],
                    [r['completeness'] for r in aggregated],
                    color=COSMIC_CONN_COLORS[model_name], marker='.',
                    label=f'cosmic-conn ({model_name}, raw data)')
        ax.plot([r['false_discovery_rate'] for r in as_aggregated],
                [r['completeness'] for r in as_aggregated],
                color=ASTROSCRAPPY_COLOR, marker='.',
                label=f'astroscrappy (objlim={CosmicRayDetector.OBJLIM:g}, '
                      f'sigfrac={CosmicRayDetector.SIGFRAC:g}, inbkg on)')
        ax.set_xscale('log')
        ax.set_xlabel('false discovery rate')
        ax.set_ylabel('completeness')
        ax.set_title('Injected real cosmic rays, scored inside the orders only')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f'\nWrote {OUTPUT_PDF}')
