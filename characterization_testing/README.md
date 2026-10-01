# Reprocessing the characterization dataset

To start running the large scale testing, run
WavelengthCalibration.ipynb notebook, the FringeFrameMaker.ipynb
notebook and then the following

```bash
python process_arcs.py --workers 8      
python process_lamp_flats.py --workers 8 --science
python fringe_correction_results.py --workers 8
python single_flat_fringe_comparison.py --workers 8
python profile_results.py --workers 8 --skip-download
python background_results.py --workers 8 --skip-download
python background_comparison.py --workers 8 --skip-download
```

`profile_results.py` writes both `trace_overlay_results.pdf` and
`profile_cross_sections.pdf` from a single reduction of each frame.

`background_results.py` writes `background_results.pdf` and a CSV of the sky
subtraction metrics defined in its module docstring. Pass `--output` and `--csv`
to keep a baseline and score a change to the background fit against it.

`background_comparison.py` fits the sky of each frame both with the pipeline's
Kelson 2003 2d fit and with the per bin fit it replaced (`per_bin_background.py`),
and writes their residuals side by side to `background_comparison.pdf`, with
in-sample and held-out metrics for both in `background_comparison.csv`.

Note that you will need to export your archive token
`export ARCHIVE_AUTH_TOKEN=your_token` to get all of the most recent data.
