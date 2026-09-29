"""Quantum acceleration interface for HPC.

Quantum computers can accelerate specific HPC workloads:
* Optimization problems (QAOA, quantum annealing)
* Linear algebra (HHL algorithm)
* Machine learning (quantum kernels)
* Simulation (quantum chemistry, materials)

This module provides:
* Quantum algorithm interfaces
* Hybrid quantum-classical orchestration
* Resource estimation
* Benchmarking against classical methods
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuantumSpecs:
    """Specifications for a quantum processing unit."""

    name: str
    num_qubits: int
    t1_us: float  # Relaxation time
    t2_us: float  # Dephasing time
    gate_time_ns: float
    readout_time_ns: float
    error_rate_1q: float
    error_rate_2q: float
    connectivity: str  # 'all_to_all', 'nearest_neighbor', 'heavy_hex'

    @property
    def coherence_time_us(self) -> float:
        """Effective coherence time."""
        return min(self.t1_us, self.t2_us)

    @property
    def max_circuit_depth(self) -> int:
        """Maximum circuit depth before decoherence."""
        return int(self.coherence_time_us * 1000 / self.gate_time_ns)


# Current and projected QPUs
IBM_EAGLE = QuantumSpecs(
    name="IBM Eagle (127 qubits)",
    num_qubits=127,
    t1_us=100,
    t2_us=150,
    gate_time_ns=50,
    readout_time_ns=500,
    error_rate_1q=1e-3,
    error_rate_2q=1e-2,
    connectivity="heavy_hex",
)

IBM_CONDOR = QuantumSpecs(
    name="IBM Condor (1121 qubits)",
    num_qubits=1121,
    t1_us=120,
    t2_us=180,
    gate_time_ns=40,
    readout_time_ns=400,
    error_rate_1q=8e-4,
    error_rate_2q=8e-3,
    connectivity="heavy_hex",
)

IONQ_HARMONY = QuantumSpecs(
    name="IonQ Harmony (32 qubits)",
    num_qubits=32,
    t1_us=10000,  # Trapped ions have long coherence
    t2_us=5000,
    gate_time_ns=1000,  # But slower gates
    readout_time_ns=10000,
    error_rate_1q=5e-4,
    error_rate_2q=5e-3,
    connectivity="all_to_all",
)


@dataclass
class QuantumKernel:
    """A quantum kernel for hybrid computing.

    Represents a quantum subroutine that can be called from
    a classical HPC application.
    """

    name: str
    qpu: QuantumSpecs
    num_qubits: int
    circuit_depth: int
    shots: int = 1024

    @property
    def execution_time_ms(self) -> float:
        """Estimated execution time in milliseconds."""
        total_gates = self.circuit_depth * self.num_qubits
        gate_time_ms = total_gates * self.qpu.gate_time_ns / 1e6
        readout_time_ms = self.qpu.readout_time_ns / 1e6
        return (gate_time_ms + readout_time_ms) * self.shots

    @property
    def success_probability(self) -> float:
        """Estimated success probability."""
        total_gates = self.circuit_depth * self.num_qubits
        error_per_gate = (self.qpu.error_rate_1q + self.qpu.error_rate_2q) / 2
        return (1 - error_per_gate) ** total_gates

    def is_feasible(self) -> bool:
        """Check if kernel is feasible on the QPU."""
        return (
            self.num_qubits <= self.qpu.num_qubits
            and self.circuit_depth <= self.qpu.max_circuit_depth
        )


class QuantumAccelerator:
    """Interface for quantum acceleration of HPC workloads.

    Provides:
    * Problem decomposition (classical + quantum)
    * Resource estimation
    * Result validation
    * Fallback to classical methods
    """

    def __init__(self, qpu: QuantumSpecs):
        self.qpu = qpu
        self.kernels: dict[str, QuantumKernel] = {}

    def register_kernel(self, kernel: QuantumKernel) -> None:
        """Register a quantum kernel."""
        self.kernels[kernel.name] = kernel

    def estimate_resources(
        self,
        problem_size: int,
        problem_type: str = "optimization",
    ) -> dict[str, Any]:
        """Estimate quantum resources for a problem.

        Parameters
        ----------
        problem_size:
            Problem size (e.g., number of variables).
        problem_type:
            'optimization', 'linear_algebra', 'simulation', 'ml'.

        Returns
        -------
        dict with resource estimates.
        """
        if problem_type == "optimization":
            # QAOA: O(n) qubits, O(p * n^2) depth
            num_qubits = problem_size
            circuit_depth = 10 * problem_size ** 2  # p=10 layers
        elif problem_type == "linear_algebra":
            # HHL: O(log n) qubits, O(poly(log n)) depth
            num_qubits = int(np.ceil(np.log2(problem_size)))
            circuit_depth = 100 * num_qubits ** 2
        elif problem_type == "simulation":
            # Trotterization: O(n) qubits, O(poly(n)) depth
            num_qubits = problem_size
            circuit_depth = 1000 * problem_size
        elif problem_type == "ml":
            # Quantum kernel: O(n) qubits, O(poly(n)) depth
            num_qubits = problem_size
            circuit_depth = 100 * problem_size
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

        kernel = QuantumKernel(
            name=f"{problem_type}_{problem_size}",
            qpu=self.qpu,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
        )

        return {
            "num_qubits": num_qubits,
            "circuit_depth": circuit_depth,
            "execution_time_ms": kernel.execution_time_ms,
            "success_probability": kernel.success_probability,
            "is_feasible": kernel.is_feasible(),
            "problem_type": problem_type,
        }

    def benchmark_vs_classical(
        self,
        problem_size: int,
        problem_type: str = "optimization",
    ) -> dict[str, Any]:
        """Benchmark quantum vs classical performance.

        Parameters
        ----------
        problem_size:
            Problem size.
        problem_type:
            Problem type.

        Returns
        -------
        dict with benchmark results.
        """
        quantum_resources = self.estimate_resources(problem_size, problem_type)

        # Classical baseline (approximate)
        if problem_type == "optimization":
            classical_time_ms = problem_size ** 3  # Brute force
        elif problem_type == "linear_algebra":
            classical_time_ms = problem_size ** 3  # Matrix inversion
        elif problem_type == "simulation":
            classical_time_ms = problem_size ** 4  # Exact diagonalization
        elif problem_type == "ml":
            classical_time_ms = problem_size ** 2  # Kernel methods
        else:
            classical_time_ms = problem_size ** 3

        quantum_time_ms = quantum_resources["execution_time_ms"]

        return {
            "problem_size": problem_size,
            "problem_type": problem_type,
            "classical_time_ms": classical_time_ms,
            "quantum_time_ms": quantum_time_ms,
            "speedup": classical_time_ms / quantum_time_ms if quantum_time_ms > 0 else 0,
            "quantum_feasible": quantum_resources["is_feasible"],
            "quantum_success_probability": quantum_resources["success_probability"],
        }


def compare_qpus(problem_size: int = 100) -> dict[str, Any]:
    """Compare different QPUs for a given problem size."""
    qpus = {
        "IBM Eagle": IBM_EAGLE,
        "IBM Condor": IBM_CONDOR,
        "IonQ Harmony": IONQ_HARMONY,
    }

    results = {}
    for name, qpu in qpus.items():
        accelerator = QuantumAccelerator(qpu)
        results[name] = accelerator.estimate_resources(problem_size, "optimization")

    return results
