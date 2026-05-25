"""Plotting utilities."""

from netfd.viz.time_domain import simulate_time_response, plot_time_response
from netfd.viz.frequency import plot_bode, plot_psi_omega, plot_identifiability_matrix
from netfd.viz.topology import plot_topology, plot_adjacency_matrix
from netfd.viz.confusion import plot_confusion_compare, plot_action_usage
from netfd.viz.training import plot_training_curves
from netfd.viz.illustrative import (
    plot_benchmark_9node_layout,
    plot_dynamics_fault, plot_signal_fault, plot_topology_fault,
    BENCHMARK_9NODE_LAYOUT,
)

__all__ = [
    "simulate_time_response", "plot_time_response",
    "plot_bode", "plot_psi_omega", "plot_identifiability_matrix",
    "plot_topology", "plot_adjacency_matrix",
    "plot_confusion_compare", "plot_action_usage",
    "plot_training_curves",
    "plot_benchmark_9node_layout",
    "plot_dynamics_fault", "plot_signal_fault", "plot_topology_fault",
    "BENCHMARK_9NODE_LAYOUT",
]
