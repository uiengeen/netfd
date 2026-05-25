"""Single-shot fault-diagnosis algorithms: probe design + classifier."""

from netfd.diagnosis.probing import (
    find_top_peaks,
    design_multi_sine_probe,
    get_peaks_for_hypotheses,
    DEFAULT_TOP_K_PEAKS,
    DEFAULT_PEAK_PROMINENCE,
    DEFAULT_PEAK_REL_THRESH,
    DEFAULT_INV_DETECT_FLOOR,
)
from netfd.diagnosis.classifier import diag_freq_weighted
from netfd.diagnosis.hypotheses import (
    build_node_fault_hypotheses,
    build_edge_fault_hypotheses,
    build_mixed_fault_hypotheses,
)

__all__ = [
    "find_top_peaks", "design_multi_sine_probe", "get_peaks_for_hypotheses",
    "DEFAULT_TOP_K_PEAKS", "DEFAULT_PEAK_PROMINENCE",
    "DEFAULT_PEAK_REL_THRESH", "DEFAULT_INV_DETECT_FLOOR",
    "diag_freq_weighted",
    "build_node_fault_hypotheses",
    "build_edge_fault_hypotheses",
    "build_mixed_fault_hypotheses",
]
