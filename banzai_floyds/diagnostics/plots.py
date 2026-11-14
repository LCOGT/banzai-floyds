"""Diagnostic plots for a series of FLOYDS spectra of the same target (RM campaign QA)."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def _as_float(column):
    return np.asarray(column, dtype=float)


def plot_line_region(spectra, redshift, lines=(4861.33, 4958.91, 5006.84), half_width=250.0, normalize_line=5006.84,
                     ax=None, title=None, cmap='viridis'):
    """Overlay many spectra around a set of rest-frame lines, each normalized to its continuum.

    ``spectra`` is a list of (wavelength, flux, label) tuples. Misregistered wavelength solutions show up as a
    horizontal spread of the narrow lines; flux-calibration scatter as a vertical spread.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    center = normalize_line * (1 + redshift)
    colors = matplotlib.colormaps[cmap](np.linspace(0, 1, max(len(spectra), 2)))
    for (wavelength, flux, label), color in zip(spectra, colors):
        wavelength, flux = np.asarray(wavelength), np.asarray(flux)
        window = np.abs(wavelength - center) < half_width
        if window.sum() < 5:
            continue
        side = window & (np.abs(wavelength - center) > 0.7 * half_width)
        norm = np.nanmedian(flux[side]) if side.sum() else np.nanmedian(flux[window])
        ax.plot(wavelength[window], flux[window] / norm, color=color, lw=0.7, alpha=0.8, label=label)
    for line in lines:
        ax.axvline(line * (1 + redshift), color='r', lw=0.6, ls='--')
    ax.set_xlabel('Observed wavelength (Å)')
    ax.set_ylabel('Flux / local continuum')
    if title:
        ax.set_title(title)
    return ax


def plot_zero_point_series(table, ax=None):
    """Wavelength zero points of object features and sky lines vs time, per order.

    Expects the columns written by the QA drivers: 'mjd', 'arm', 'o3_shift', 'o3_shift_err', 'sky5577_shift',
    'tellA_shift'.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    for arm, color in [('blue', 'tab:blue'), ('red', 'tab:red')]:
        rows = table[table['arm'] == arm]
        mjd = _as_float(rows['mjd'])
        ax.errorbar(mjd, _as_float(rows['o3_shift']), _as_float(rows['o3_shift_err']), fmt='o', color=color,
                    ms=4, label=f'[OIII] {arm} order (object)')
        ax.plot(mjd, _as_float(rows['sky5577_shift']), 's', mfc='none', color=color, ms=5,
                label=f'[OI]5577 sky, {arm} order')
        if arm == 'red' and 'tellA_shift' in rows.colnames:
            ax.plot(mjd, _as_float(rows['tellA_shift']), '^', color='k', ms=5, label='telluric A band (object)')
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_xlabel('MJD')
    ax.set_ylabel('Measured − expected (Å)')
    ax.legend(fontsize=8, ncol=2)
    return ax


def plot_shift_correlations(table, axes=None):
    """Scatter of the red-order [OIII] shift against the telluric A band, the blue [OIII] and the sky shifts.

    Object features that move together but not with the sky are the signature of the target position across a
    wide slit (or of an object-specific flexure), which arc and sky calibrations cannot correct.
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 4))
    red = table[table['arm'] == 'red']
    blue = table[table['arm'] == 'blue']
    pairs = [(_as_float(red['tellA_shift']), 'telluric A shift, red (Å)'),
             (_as_float(blue['o3_shift']), '[OIII] shift, blue (Å)'),
             (_as_float(red['sky5577_shift']), 'sky 5577 shift, red (Å)')]
    y = _as_float(red['o3_shift'])
    for ax, (x, label) in zip(axes, pairs):
        good = np.isfinite(x) & np.isfinite(y)
        ax.plot(x[good], y[good], 'o', ms=4)
        if good.sum() > 2:
            r = np.corrcoef(x[good], y[good])[0, 1]
            ax.set_title(f'r = {r:.2f}')
        ax.set_xlabel(label)
        ax.set_ylabel('[OIII] shift, red (Å)')
    return axes


def plot_line_flux_series(table, ax=None):
    """Narrow-line flux, normalized to its median, vs time, per order. It should be constant for RM."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    for arm, color in [('blue', 'tab:blue'), ('red', 'tab:red')]:
        rows = table[table['arm'] == arm]
        flux = _as_float(rows['o3_flux'])
        median = np.nanmedian(flux)
        ax.errorbar(_as_float(rows['mjd']), flux / median, _as_float(rows['o3_flux_err']) / median, fmt='o',
                    color=color, ms=4, label=f'{arm} order')
    ax.axhline(1, color='0.5', lw=0.5)
    ax.set_xlabel('MJD')
    ax.set_ylabel('[OIII]5007 flux / median')
    ax.legend(fontsize=8)
    return ax
