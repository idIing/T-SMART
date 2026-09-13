import numpy as np
import scipy.stats as stats
import itertools
from tsqa.tools.basic_stats import _clean_1d

class TSAQATransformation:
    """
    Deterministic tools for TSAQA's Data Transformation task category.
    Simulates signal processing and mathematical transformations on 1-D series.
    """

    @staticmethod
    def shift_offset(ts, offset: float) -> np.ndarray:
        """Add a constant offset to the series: y_t = x_t + c."""
        arr = _clean_1d(ts)
        return arr + offset

    @staticmethod
    def scale(ts, scale_factor: float) -> np.ndarray:
        """Multiply the series by a scaling factor: y_t = c * x_t."""
        arr = _clean_1d(ts)
        return arr * scale_factor

    @staticmethod
    def lag(ts, shift_steps: int, fill_value=np.nan) -> np.ndarray:
        """Shift the series temporally: y_t = x_{t - shift}."""
        arr = np.asarray(ts, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("Time series must be a non-empty 1-D array.")
        
        out = np.empty_like(arr)
        if shift_steps > 0:
            out[:shift_steps] = fill_value
            out[shift_steps:] = arr[:-shift_steps]
        elif shift_steps < 0:
            out[shift_steps:] = fill_value
            out[:shift_steps] = arr[-shift_steps:]
        else:
            out[:] = arr
        return out

    @staticmethod
    def moving_average(ts, window_size: int) -> np.ndarray:
        """Apply a simple moving average filter of window_size."""
        arr = _clean_1d(ts)
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if window_size > arr.size:
            raise ValueError(f"window_size ({window_size}) cannot exceed series size ({arr.size}).")
        
        # Valid convolution padding or standard rolling window
        ret = np.cumsum(arr, dtype=float)
        ret[window_size:] = ret[window_size:] - ret[:-window_size]
        return ret[window_size - 1:] / window_size

    @staticmethod
    def difference(ts, order: int = 1) -> np.ndarray:
        """Compute the difference of order: y_t = x_t - x_{t-order}."""
        arr = _clean_1d(ts)
        if order <= 0:
            raise ValueError("difference order must be a positive integer.")
        if order >= arr.size:
            raise ValueError(f"difference order ({order}) cannot exceed series size ({arr.size}).")
        return np.diff(arr, n=order)

    @staticmethod
    def detrend(ts) -> np.ndarray:
        """Remove a linear OLS trend from the time series."""
        arr = _clean_1d(ts)
        t = np.arange(arr.size, dtype=float)
        # Fit OLS line: y = alpha + beta * t
        beta, alpha, _, _, _ = stats.linregress(t, arr)
        trend = alpha + beta * t
        return arr - trend


class TSAQAPuzzler:
    """
    Deterministic solver for TSAQA's Puzzling (PZ) format.
    Reconstructs the original time series from scrambled segments.
    """

    @staticmethod
    def junction_cost(seg1: np.ndarray, seg2: np.ndarray) -> float:
        """
        Evaluate the discontinuity cost at the junction if seg1 is followed by seg2.
        Combines point value difference and first-order slope difference.
        """
        s1 = _clean_1d(seg1)
        s2 = _clean_1d(seg2)
        if s1.size < 2 or s2.size < 2:
            return float((s1[-1] - s2[0]) ** 2)
        
        val_diff = float((s1[-1] - s2[0]) ** 2)
        # Continuity of slope: (s1[-1] - s1[-2]) vs (s2[1] - s2[0])
        slope1 = s1[-1] - s1[-2]
        slope2 = s2[1] - s2[0]
        slope_diff = float((slope1 - slope2) ** 2)
        
        return val_diff + 0.5 * slope_diff

    @classmethod
    def solve_jigsaw(cls, segments: list, reference: np.ndarray = None) -> list:
        """
        Solve the segment reconstruction puzzle.
        
        Args:
            segments: List of 1-D arrays (scrambled parts of the series).
            reference: Optional reference 1-D array of the complete series.
            
        Returns:
            list[int]: The list of indices in `segments` indicating the correct sequence order.
        """
        clean_segs = [_clean_1d(s) for s in segments]
        n_segs = len(clean_segs)
        if n_segs == 0:
            return []

        # If a reference is provided, match each segment to the best-fitting window of the reference
        if reference is not None:
            ref = _clean_1d(reference)
            best_positions = []
            for seg in clean_segs:
                seg_len = seg.size
                if seg_len >= ref.size:
                    best_positions.append(0.0)
                    continue
                # Slide window and find minimum DTW / Euclidean distance
                min_dist = float("inf")
                best_idx = 0
                for start in range(ref.size - seg_len + 1):
                    ref_window = ref[start:start+seg_len]
                    dist = float(np.mean((seg - ref_window) ** 2))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = start
                best_positions.append(best_idx)
            
            # Sort segment indices based on their mapped starting positions in the reference
            return [idx for idx, _ in sorted(enumerate(best_positions), key=lambda x: x[1])]

        # If no reference is provided, solve as a Traveling Salesperson Problem (TSP)
        # by searching all permutations to minimize total boundary junction cost.
        best_perm = list(range(n_segs))
        min_total_cost = float("inf")
        
        for perm in itertools.permutations(range(n_segs)):
            total_cost = 0.0
            for i in range(n_segs - 1):
                total_cost += cls.junction_cost(clean_segs[perm[i]], clean_segs[perm[i+1]])
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_perm = list(perm)
                
        return best_perm
