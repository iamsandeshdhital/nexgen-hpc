"""HPC benchmark suite with real algorithm benchmarks.

Benchmarks:
* Interconnect: routing, wavelength assignment, topology optimization
* Memory: data placement, migration, bandwidth allocation
* Scheduler: DVFS, backfill, power capping
* Accelerator: roofline, load balancing, offloading
* In-situ: compression, feature extraction
* Fault tolerance: checkpointing, recovery, prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .accelerator import AcceleratorManager, KernelProfile, NVIDIA_H100
from .fault_tolerance import FaultToleranceManager, estimate_fault_tolerance_savings
from .hybrid import NextGenSystem, HPCJob, NextGenOrchestrator, create_nexgen_system
from .insitu import InSituAnalyzer, estimate_io_savings
from .interconnect import compare_interconnects, benchmark_routing_algorithms, create_fat_tree_topology, PHOTONIC_WDM
from .memory import compare_memory_technologies, benchmark_memory_tiers, MemoryHierarchy, MemoryTier, DataObject, HBM3E
from .scheduler import EnergyAwareScheduler, NodeSpecs, Job, benchmark_scheduler, optimize_frequency_sweep


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    metrics: dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()


class HPCBenchmark:
    """Comprehensive HPC benchmark suite."""

    def __init__(self, system: NextGenSystem | None = None):
        self.system = system or create_nexgen_system()
        self.results: list[BenchmarkResult] = []

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        self.results = []
        self.results.append(self.benchmark_interconnect())
        self.results.append(self.benchmark_memory())
        self.results.append(self.benchmark_accelerator())
        self.results.append(self.benchmark_insitu())
        self.results.append(self.benchmark_energy())
        self.results.append(self.benchmark_fault_tolerance())
        self.results.append(self.benchmark_scalability())
        return self.results

    def benchmark_interconnect(self) -> BenchmarkResult:
        """Benchmark interconnect with real algorithms."""
        # Compare technologies
        tech_results = compare_interconnects(data_size_mb=1000, num_nodes=1024)

        # Benchmark routing algorithms
        routing_results = benchmark_routing_algorithms(num_nodes=256)

        # Benchmark topology optimization
        topology = create_fat_tree_topology(k=64, link_specs=PHOTONIC_WDM)
        wavelength_assignment = topology.wavelength_assignment(num_wavelengths=8)

        return BenchmarkResult(
            name="interconnect",
            metrics={
                "technologies": tech_results,
                "routing": routing_results,
                "topology": {
                    "num_nodes": topology.num_nodes,
                    "num_links": len(topology.link_capacities) // 2,
                    "wavelengths_used": len(set(wavelength_assignment.values())) if wavelength_assignment else 0,
                },
            },
        )

    def benchmark_memory(self) -> BenchmarkResult:
        """Benchmark memory with real algorithms."""
        # Compare technologies
        tech_results = compare_memory_technologies()

        # Benchmark memory tiers
        tier_results = benchmark_memory_tiers(data_size_gb=1.0)

        # Benchmark data placement
        tiers = [
            MemoryTier(HBM3E, num_stacks=4, distance_ns=10),
        ]
        hierarchy = MemoryHierarchy(tiers)

        data_objects = [
            DataObject(name="hot_data", size_gb=10, access_frequency=100, read_write_ratio=0.8, locality=0.9),
            DataObject(name="warm_data", size_gb=100, access_frequency=10, read_write_ratio=0.5, locality=0.5),
            DataObject(name="cold_data", size_gb=1000, access_frequency=1, read_write_ratio=0.2, locality=0.1),
        ]

        placement_result = hierarchy.optimize_placement(data_objects)

        return BenchmarkResult(
            name="memory",
            metrics={
                "technologies": tech_results,
                "tiers": tier_results,
                "placement": placement_result,
            },
        )

    def benchmark_accelerator(self) -> BenchmarkResult:
        """Benchmark accelerator with real algorithms."""
        manager = AcceleratorManager()

        # Compare accelerators
        accel_results = manager.compare_accelerators()

        # Benchmark kernel offloading
        kernels = [
            KernelProfile(name="matmul", arithmetic_intensity=100, parallelism=100000, data_size_gb=10, control_flow_complexity=0.1),
            KernelProfile(name="conv", arithmetic_intensity=50, parallelism=50000, data_size_gb=5, control_flow_complexity=0.3),
            KernelProfile(name="reduction", arithmetic_intensity=10, parallelism=1000, data_size_gb=1, control_flow_complexity=0.5),
        ]

        offload_results = []
        for kernel in kernels:
            rec = manager.recommend_accelerator(kernel)
            offload_results.append({
                "kernel": kernel.name,
                "recommended": rec["recommended_accelerator"],
                "speedup": rec["speedup_vs_cpu"],
            })

        # Benchmark load balancing
        load_balance = manager.load_balance(kernels, num_accelerators=4)

        return BenchmarkResult(
            name="accelerator",
            metrics={
                "accelerators": accel_results,
                "offloading": offload_results,
                "load_balancing": load_balance,
            },
        )

    def benchmark_insitu(self) -> BenchmarkResult:
        """Benchmark in-situ analysis with real algorithms."""
        # Create test data
        data = np.random.rand(100, 100, 100)
        analyzer = InSituAnalyzer()

        # Run all methods
        results = analyzer.analyze_pipeline(data)

        # Benchmark adaptive reduction
        adaptive_result = analyzer.adaptive_reduction(data, target_ratio=0.9)

        # Estimate I/O savings
        io_savings = estimate_io_savings(
            data_size_gb=data.nbytes / 1e9,
            reduction_ratio=0.9,
        )

        return BenchmarkResult(
            name="insitu",
            metrics={
                "methods": {
                    name: {
                        "compression_ratio": r.compression_ratio,
                        "information_loss": r.information_loss,
                    }
                    for name, r in results.items()
                },
                "adaptive": {
                    "compression_ratio": adaptive_result.compression_ratio,
                    "information_loss": adaptive_result.information_loss,
                    "method": adaptive_result.method,
                },
                "io_savings": io_savings,
            },
        )

    def benchmark_energy(self) -> BenchmarkResult:
        """Benchmark energy with real algorithms."""
        # Benchmark scheduler
        scheduler_results = benchmark_scheduler(num_jobs=50, num_nodes=256)

        # Benchmark DVFS optimization
        node = NodeSpecs(
            name="test",
            num_cores=128,
            memory_gb=512,
            tdp_w=500,
            base_freq_ghz=2.0,
            max_freq_ghz=3.5,
            idle_power_w=100,
        )
        job = Job(
            name="test",
            num_nodes=1,
            estimated_runtime_s=3600,
            estimated_energy_kwh=100,
        )
        dvfs_result = optimize_frequency_sweep(job, node)

        return BenchmarkResult(
            name="energy",
            metrics={
                "scheduler": scheduler_results,
                "dvfs": {
                    "optimal_frequency_ghz": dvfs_result["optimal_frequency_ghz"],
                    "energy_savings_percent": dvfs_result["energy_savings_percent"],
                },
            },
        )

    def benchmark_fault_tolerance(self) -> BenchmarkResult:
        """Benchmark fault tolerance with real algorithms."""
        savings = estimate_fault_tolerance_savings(
            job_runtime_hours=24,
            mtbf_hours=12,
            checkpoint_interval_minutes=5,
            recovery_time_minutes=2,
        )

        manager = FaultToleranceManager()
        optimization = manager.optimize_checkpoint_interval(
            job_runtime_s=24 * 3600,
            mtbf_s=12 * 3600,
            checkpoint_time_s=60,
        )

        return BenchmarkResult(
            name="fault_tolerance",
            metrics={
                "savings": savings,
                "optimization": optimization,
            },
        )

    def benchmark_scalability(self) -> BenchmarkResult:
        """Benchmark scalability."""
        sizes = [256, 512, 1024, 2048]
        results = {}

        for size in sizes:
            system = create_nexgen_system(num_nodes=size)
            job = HPCJob(
                name=f"scale_{size}",
                num_nodes=size,
                estimated_runtime_s=3600,
                estimated_energy_kwh=100,
                priority=5,
            )
            orchestrator = NextGenOrchestrator(system)
            result = orchestrator.submit_job(job)
            results[size] = {
                "execution_time_s": result.get("execution_time_s", 0),
                "energy_kwh": result.get("estimated_energy_kwh", 0),
            }

        return BenchmarkResult(
            name="scalability",
            metrics={"scaling": results},
        )

    def save_results(self, results: list[BenchmarkResult], path: str) -> None:
        """Save benchmark results to JSON."""
        import json
        from dataclasses import asdict

        data = [asdict(r) for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def run_full_benchmark_suite(
    system: NextGenSystem | None = None,
) -> dict[str, Any]:
    """Run the full benchmark suite."""
    benchmark = HPCBenchmark(system)
    results = benchmark.run_all()

    return {
        "num_benchmarks": len(results),
        "benchmarks": [
            {
                "name": r.name,
                "metrics": r.metrics,
                "timestamp": r.timestamp,
            }
            for r in results
        ],
    }
