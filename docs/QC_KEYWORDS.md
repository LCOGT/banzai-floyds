# Quality keywords (L1Q*)

Every science and standard product carries per-frame quality keywords in the primary header of the `-1d` file and
the `SCI` header of the `-2d` file. The pipeline writes them in its last stage (`banzai_floyds.quality.QualityMetrics`);
for products reduced before that stage existed, `banzai_floyds_qc` computes the same values from the files:

```
banzai_floyds_qc PROCESSED_DIR [PROCESSED_DIR ...] --table qc.csv           # dry run + table
banzai_floyds_qc PROCESSED_DIR [PROCESSED_DIR ...] --write --table qc.csv   # also write the headers
```

The table has one row per product with the observing conditions, the pipeline's own L1 keywords and all L1Q
keywords, ready to join to a master table on the file name. `L1QVER` gives the version of this scheme; a change of
a definition or threshold increases it.

## Observing conditions (copied from the raw frame by LCO)

| Keyword | Meaning |
|---|---|
| `AGFWHM` | Seeing from the guide camera (arcsec); missing or 0 in some frames, see `L1QSEE1/2` |
| `AIRMASS` | Airmass |
| `MOONFRAC`, `MOONDIST`, `MOONALT`, `MOONSTAT` | Moon illumination, distance (deg), altitude, up/down |
| `WMSCLOUD` | Sky minus ambient temperature (deg C): above about -30 is cloudy |
| `WMSSKYBR` / `SKYMAG` | Sky brightness (mag/arcsec^2) |
| `WMSHUMID`, `WINDSPEE`, `SUNALT`, `AGSTATE` | Humidity, wind, Sun altitude, guiding state |

## Pipeline keywords used by the flags

| Keyword | Meaning |
|---|---|
| `L1TRCMOD`, `L1TRCADR` | Object-trace model (`adr` = physical model) and its refraction scale |
| `L1FWHSRC`, `L1SHPSRC` | Source of the profile width / shape per order: `fit`, `star` (stored), `gaussian` |
| `L1BKNK1/2`, `L1BKMW1/2` | Sky-spline knots; half width (profile sigma) of the object mask in the sky fit |
| `L1FRGRES` | Fringe residual ratio after the harmonic fringe fit |
| `L1FLXNST`, `L1FLXGRY` | Standards in the flux ensemble; max/min grey factor among them |
| `L1WZP1/2` | Telluric wavelength zero point applied (A) |
| `L1OBJDET` | Object detected |

## Cosmic rays checked against the other exposure (`visit_cosmics`)

| Keyword | Meaning | Normal range (OGG / COJ, 106 frames) |
|---|---|---|
| `L1CRMETH` | `pair` (checked against the partner) or `lacosmic` (no partner: LA Cosmic with the object model) | standards `lacosmic` |
| `L1CRPART` | File name of the partner exposure | |
| `L1CRNLA` | Pixels flagged by LA Cosmic | 115-715 / 175-1755 |
| `L1CRNPX` | Pixels flagged at the end | 80 % / 90 % of `L1CRNLA` |
| `L1CRKEEP`, `L1CRDROP` | LA groups confirmed / unflagged by the partner | 1-35, 4-28 / 6-86, 4-32 |
| `L1CRUNCH` | LA groups kept because the partner cannot check them (masked, order edges, extended difference, flagged in both exposures off the trace: warm pixels) | 5-62 / 10-76 |
| `L1CRADD` | Groups added (compact strong hits LA Cosmic missed) | 0-6 / 0-3 |
| `L1CRMISM` | Fraction of the orders where the matched partner differs over an extended area (partner cannot judge) | < 0.06 / < 0.025 |
| `L1CRSHAR` | LA pixels on the trace also flagged in the partner, judged against the partner's own values | |
| `L1CRDX1`, `L1CRDX2` | Offset of the partner along the dispersion per order (px) | < 0.6; COJ blue order unreliable (up to 3.4) |

## L1Q keywords

| Keyword | Meaning | Normal range (COJ/OGG final v3) |
|---|---|---|
| `L1QSN1`, `L1QSN2` | Median S/N per bin, red (6000-8800 A) / blue (3800-5400 A) order | standards 50-350, AGN 90-220 |
| `L1QXJ1`, `L1QXJ2` | Extraction scatter / errors: robust rms of the second difference of the extracted flux over its expected error (1 = noise; insensitive to a smooth spectrum, raised by real structure at high S/N) | 1.0-2.1 |
| `L1QBJ1`, `L1QBJ2` | Same for the sky model under the object, over the flux errors | < 0.6 |
| `L1QCR1`, `L1QCR2` | % of trace-core pixels (within one FWHM of the trace) flagged as cosmic rays | < 0.6 % |
| `L1QCRLN`, `L1QCRLP` | Cosmic-ray flags on the trace core of strong narrow emission lines (bins > 1.25 x a 15-bin running median and 10 sigma above it), and the number of such core pixels | 0-2 |
| `L1QCRB1`, `L1QCRB2` | Bins masked (EXTRACTED mask bit 4) because cosmic-ray flags removed > 50 % of their profile weight; only in products from the new extraction | 0 |
| `L1QSEE1`, `L1QSEE2` | Measured FWHM of the object along the slit (arcsec) at 7000 / 4500 A: the seeing in the slit | 1.4-4 |
| `L1QPOFF` | Object offset along the slit from the order centre (arcsec, red order) | +-2.3 |
| `L1QSTDGR` | Standards only: light collected relative to the clear ensemble of the other standards (S_ensemble / S_self, 5500-6500 A); low = clouds or a missed acquisition | 0.6-1.3 |
| `L1QSTDDT`, `L1QSTDSP` | Science only: days to the nearest / farthest standard in the flux ensemble | 0.1-6 |
| `L1QFLAG` | Bitmask of the flags below | 0 |
| `L1QFLAGS` | Names of the flags set, comma separated, or `OK` | `OK` |

## Flags (`L1QFLAG` bits)

| Bit | Name | Set when | What it found in final v3 |
|---|---|---|---|
| 1 | `XJIT` | `L1QXJ1` or `L1QXJ2` > 2.5 | COJ Fairall 9 under moon and poor seeing (sky spline with a knot per pixel); OGG BD+28 4211 2025-08-19 (star core flagged as cosmic rays) |
| 2 | `BKJIT` | `L1QBJ1` or `L1QBJ2` > 1.0 | the same COJ frames: the sky model followed the noise |
| 4 | `CRCORE` | `L1QCR1` or `L1QCR2` > 1 % | OGG BD+28 4211 in good seeing: the core of a bright star taken for cosmic rays |
| 8 | `CRLINE` | `L1QCRLN` >= 3 | NGC 4151 Halpha clipped as cosmic rays (one frame 2.6 x too bright) |
| 16 | `LOWSNR` | `L1QSN1` or `L1QSN2` < 10 | |
| 32 | `TRACE` | `L1TRCMOD` not `adr`, or abs(`L1TRCADR`) > 3 | GD 153 2025-05-25 (S/N 6, refraction scale -20) |
| 64 | `PROFILE` | profile width not fitted (`L1FWHSRC` not `fit`) | |
| 128 | `BKMASK` | `L1BKMW1` or `L1BKMW2` < 6 | wide object, little sky left beside it |
| 256 | `STDGREY` | `L1QSTDGR` < 0.5 | clouded or missed standards (GD 153 2025-05-25: 0.09; BD+28 4211 2025-08-10: 0.06) |
| 512 | `STDFAR` | `L1QSTDDT` > 5 days | science far from any standard (early NGC 4151; Fairall 9 between Feige 110 visits) |
| 1024 | `FRINGE` | `L1FRGRES` > 0.75 | |
| 2048 | `NOOBJ` | `L1OBJDET` false | |
| 4096 | `SEEING` | `L1QSEE1` or `L1QSEE2` > 1.5 x slit width | slit losses that depend on wavelength |
| 8192 | `CRBIN` | `L1QCRB1` or `L1QCRB2` > 0 | |

Thresholds were set from the distributions of the 317 science and standard products of the final v3 reductions
of COJ and OGG, just outside the normal range above, and checked against the frames found by hand (RM report).
Flags mark frames to inspect or exclude; they do not change the reduction.
