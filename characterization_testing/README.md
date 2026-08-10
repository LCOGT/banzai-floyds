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
python adr_trace_validation.py --workers 8 --skip-download
```

Three of these characterize the profile fitting on real data. `trace_overlay_results.py` draws the
fitted trace over each order of every science frame, and `profile_cross_sections.py` measures how
well the fitted Gaussian describes the stacked cross section, including whether an extended host
warrants a second component; both write a CSV alongside the PDF, and the PDF's last page is the
summary over all frames.

`adr_trace_validation.py` asks whether atmospheric differential refraction explains the shape of the
trace, which would let the free degree 5 polynomial be replaced by a curve predicted from the header.
It does not: refraction is real and removes 34% of the chi^2 against a straight line on every order
where the slit is vertical, but a free polynomial is still demanded on 81% of orders, and what is
left over is per-frame drift rather than an instrumental shape a template could hold. The trace
keeps its polynomial; the script is the record of why. It writes a CSV only, no PDF.

Note that you will need to export your archive token
`export ARCHIVE_AUTH_TOKEN=your_token` to get all of the most recent data.
