"""Re-reduce the raw FLOYDS arc frames already on disk (test_data/raw), skipping the archive query.

Use this to regenerate the a91 products after a pipeline change (e.g. the degree-5 red wavelength
fit) without re-downloading. Reuses the worker setup and pipeline driver from make_residuals_pdf.py.

    python reprocess_raw.py --workers 4
"""
import argparse
import os
from glob import glob

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

from make_residuals_pdf import process_frames, bound_arcs_to_skyflat_windows, RAW_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--raw-dir', default=RAW_DIR)
    args = parser.parse_args()

    paths = sorted(glob(os.path.join(args.raw_dir, '*.fits.fz')))
    print(f'Re-processing {len(paths)} raw frames from {args.raw_dir}')
    process_frames(paths, args.workers)
    # Tie each arc's validity window to its skyflat (order solution) epoch so an arc from before the
    # orders moved is never picked as the initial solution for a later arc.
    bound_arcs_to_skyflat_windows()
