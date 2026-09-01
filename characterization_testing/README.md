# Reprocessing the characterization dataset

To start running the large scale testing, run
WavelengthCalibration.ipynb notebook, the FringeFrameMaker.ipynb
notebook and then the following

```bash
python process_arcs.py --workers 8      
python process_lamp_flats.py --workers 8 --science
python fringe_correction_results.py --workers 8
python single_flat_fringe_comparison.py --workers 8
python trace_overlay_results.py --workers 8 --skip-download
python profile_cross_sections.py --workers 8 --skip-download
```

Note that you will need to export your archive token
`export ARCHIVE_AUTH_TOKEN=your_token` to get all of the most recent data.
