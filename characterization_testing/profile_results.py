"""Build the trace overlay and the profile cross section reports from a single pass over the data.

Run from the characterization_testing directory after the WavelengthCalibration.ipynb setup cells
have created test_data/test.db (and ideally after process_arcs.py and process_lamp_flats.py so the
arcs and fringe masters are in the calibration database):
"""
import argparse
import os

os.environ['OPENTSDB_PYTHON_METRICS_TEST_MODE'] = 'True'
os.environ.setdefault('DB_ADDRESS', 'sqlite:///test_data/test.db')

import matplotlib
matplotlib.use('Agg')

import trace_overlay_results
import profile_cross_sections
from process_arcs import RAW_DIR
from process_lamp_flats import make_context
from reduction_utils import reduce_to_stage, frame_metadata, clear_cosmic_ray_flags
from report_utils import raw_frame_paths, write_reports

_context = None


def _init_worker():
    """Build a banzai context once per worker process."""
    global _context
    _context = make_context()


def profile_frame(path: str) -> tuple:
    """Reduce one raw e00 frame and take both the trace overlay and the cross section measurements.

    Returns (path, record, error), where the record carries the part each report draws under its own
    key. Both measurements need the fitted profile, so a frame with no detected object is skipped by
    both.
    """
    try:
        image, note, error = reduce_to_stage(path, _context, 'banzai_floyds.extract.Extractor')
        if error is not None:
            return path, None, error
        clear_cosmic_ray_flags(image, _context)
        if image.profile_fits is None:
            return path, None, 'no object was detected in the slit'
        record = {'metadata': frame_metadata(path, image), 'note': note,
                  'trace': trace_overlay_results.trace_record(image),
                  'cross_sections': profile_cross_sections.cross_section_record(image)}
        return path, record, None
    except Exception as e:
        return path, None, str(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel pipeline workers (default 4)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip the archive query and just use the e00s already on disk')
    parser.add_argument('--glob', default=os.path.join(RAW_DIR, '*e00*'),
                        help='Glob for the raw e00 frames to reduce')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to write the PDFs and CSVs into')
    args = parser.parse_args()

    reports = [module.make_report(os.path.join(args.output_dir, module.OUTPUT_PDF),
                                  os.path.join(args.output_dir, module.OUTPUT_CSV))
               for module in [trace_overlay_results, profile_cross_sections]]
    paths = raw_frame_paths(args.glob, args.skip_download)
    write_reports(paths, profile_frame, reports, workers=args.workers, initializer=_init_worker)
