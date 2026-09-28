"""Drive a set of raw frames through a per frame measurement and write out the PDF and CSV.

Every characterization script runs the same loop: reduce the frames in a process pool, write the CSV
rows for each one as it arrives, draw its page into a PDF, and finish with a summary page over all
of the rows. The loop lives here so that a frame function that measures several different things
from a single reduction (profile_results.py) can feed all of their reports in one pass.
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from glob import glob
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from process_arcs import download_frames
from reduction_utils import frame_title


def raw_frame_paths(pattern: str, skip_download: bool) -> list:
    """The raw frames to run over, downloading the archive query windows first unless told not to."""
    if not skip_download:
        if 'ARCHIVE_AUTH_TOKEN' not in os.environ:
            print('ARCHIVE_AUTH_TOKEN is not set; skipping the archive download', file=sys.stderr)
        else:
            download_frames('e00')
    paths = sorted(glob(pattern), key=os.path.basename)
    if not paths:
        raise SystemExit(f'No raw science frames match {pattern}.')
    return paths


@dataclass
class Report:
    """One PDF and CSV pair, and the functions that fill them.

    `key` names the part of the per frame record this report draws. A frame function returns
    {'metadata': ..., 'note': ..., <key>: ...} for each report it feeds, and the shared metadata and
    note are merged back into each report's record so the plotting and CSV functions see the same
    flat record they would get from a script that measures only one thing.
    """
    key: str
    pdf: str
    csv: str
    fields: list
    csv_rows: Callable
    plot_frame: Callable
    plot_summary: Callable
    summary_title: str
    figsize: tuple = (11, 8.5)


def write_reports(paths: list, frame_function: Callable, reports: list, workers: int = 4,
                  initializer: Callable = None) -> None:
    """Reduce every frame in a pool and write a page and CSV rows per report as each one arrives.

    Drawing each page as it arrives rather than holding hundreds of them in memory is what puts the
    summary on the last page.
    """
    rows = {report.key: [] for report in reports}
    n_pages = 0
    with ExitStack() as stack:
        csv_files, writers, pdfs = {}, {}, {}
        for report in reports:
            csv_files[report.key] = stack.enter_context(open(report.csv, 'w', newline=''))
            writers[report.key] = csv.DictWriter(csv_files[report.key], fieldnames=report.fields,
                                                 restval='', extrasaction='ignore')
            writers[report.key].writeheader()
            pdfs[report.key] = stack.enter_context(PdfPages(report.pdf))

        with ProcessPoolExecutor(max_workers=workers, initializer=initializer) as pool:
            for path, record, error in pool.map(frame_function, paths):
                filename = os.path.basename(path)
                if error is not None:
                    print(f'Skipping {filename}: {error}', file=sys.stderr)
                for report in reports:
                    sub = None
                    if record is not None:
                        sub = dict(record[report.key], metadata=record['metadata'], note=record['note'])
                    for row in report.csv_rows(sub, error, filename):
                        writers[report.key].writerow(row)
                        rows[report.key].append(row)
                    csv_files[report.key].flush()
                    if sub is None:
                        continue
                    fig = plt.figure(figsize=report.figsize)
                    title = frame_title(record['metadata'])
                    if record['note']:
                        title += f'  [{record["note"]}]'
                    fig.suptitle(title, fontsize=9)
                    report.plot_frame(fig, sub)
                    pdfs[report.key].savefig(fig)
                    plt.close(fig)
                if record is not None:
                    n_pages += 1

        if n_pages == 0:
            raise SystemExit('No frames were successfully reduced.')
        for report in reports:
            fig = plt.figure(figsize=report.figsize)
            fig.suptitle(report.summary_title, fontsize=10)
            report.plot_summary(fig, rows[report.key])
            pdfs[report.key].savefig(fig)
            plt.close(fig)

    for report in reports:
        print(f'Wrote {report.pdf} ({n_pages} frames, summary on the last page) and {report.csv}')
