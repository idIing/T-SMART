import numpy as np
from scipy.signal import detrend


def fft_period(ts: np.ndarray) -> dict:
    """
    Estimate dominant periodicity using the real FFT magnitude spectrum.

    Applies linear detrending (not just mean subtraction) before the FFT so
    that a strong linear or exponential trend does not dominate the spectrum
    and produce a spurious dominant_period equal to the series length.

    Args:
        ts: 1-D time series array

    Returns:
        dominant_period: 1 / dominant_freq in samples; None if dominant_freq == 0
        dominant_freq:   frequency (cycles/sample) with highest magnitude, DC excluded
        spectrum_top3:   list of up to 3 (freq, magnitude) tuples for strongest peaks
        spectral_entropy: float in [0, 1] — 0 = single pure tone, 1 = flat spectrum.
                          Low values indicate a strongly periodic signal.

    Used by: periodicity branch (global + local), similarity branch
    """
    ts_f = detrend(ts.astype(float), type="linear")

    magnitudes = np.abs(np.fft.rfft(ts_f))
    freqs      = np.fft.rfftfreq(len(ts_f))

    # zero out DC bin
    magnitudes[0] = 0.0

    # spectral entropy — normalise magnitudes to a probability distribution
    mag_sum = magnitudes.sum()
    if mag_sum > 0:
        p = magnitudes / mag_sum
        p_nonzero = p[p > 0]
        spectral_entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)) /
                                  np.log(len(magnitudes)))
    else:
        spectral_entropy = 1.0

    top_idx       = np.argsort(magnitudes)[::-1][:3]
    spectrum_top3 = [(float(freqs[i]), float(magnitudes[i])) for i in top_idx]

    dominant_idx    = top_idx[0]
    dominant_freq   = float(freqs[dominant_idx])
    dominant_period = (1.0 / dominant_freq) if dominant_freq > 0 else None

    return {
        "dominant_period":  dominant_period,
        "dominant_freq":    dominant_freq,
        "spectrum_top3":    spectrum_top3,
        "spectral_entropy": spectral_entropy,
    }
