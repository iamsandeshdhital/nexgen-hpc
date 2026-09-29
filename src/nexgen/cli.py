"""Command-line interface for the NextGen HPC Framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .accelerator import AcceleratorManager, KernelProfile
from .benchmark import run_full_benchmark_suite
from .fault_tolerance import estimate_fault_tolerance_savings, FaultToleranceManager
from .hybrid import create_nexgen_system, HPCJob, NextGenOrchestrator
from .insitu import InSituAnalyzer, estimate_io_savings
from .interconnect import compare_interconnects, benchmark_routing_algorithms, create_fat_tree_topology, PHOTONIC_WDM
from .memory import compare_memory_technologies, benchmark_memory_tiers, MemoryHierarchy, MemoryTier, DataObject, HBM3E
from .scheduler import EnergyAwareScheduler, NodeSpecs, Job, benchmark_scheduler, optimize_frequency_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexgen-run",
        description="NextGen HPC Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"nexgen {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("--output", type=Path, default=None)

    # Interconnect command
    interconnect_parser = subparsers.add_parser("interconnect", help="Compare interconnects")
    interconnect_parser.add_argument("--nodes", type=int, default=1024)
    interconnect_parser.add_argument("--data-mb", type=float, default=1000)

    # Memory command
    memory_parser = subparsers.add_parser("memory", help="Compare memory technologies")

    # Accelerator command
    accelerator_parser = subparsers.add_parser("accelerator", help="Compare accelerators")

    # In-situ command
    insitu_parser = subparsers.add_parser("insitu", help="Test in-situ analysis")

    # Fault tolerance command
    fault_parser = subparsers.add_parser("fault", help="Estimate fault tolerance savings")

    # Job command
    job_parser = subparsers.add_parser("submit", help="Submit HPC job")
    job_parser.add_argument("--nodes", type=int, default=1024)
    job_parser.add_argument("--gpus", type=int, default=0)
    job_parser.add_argument("--tpus", type=int, default=0)
    job_parser.add_argument("--data-gb", type=float, default=100)
    job_parser.add_argument("--intensity", type=float, default=100)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        print("Running full benchmark suite...")
        results = run_full_benchmark_suite()
        print(json.dumps(results, indent=2, default=str))
        if args.output:
            args.output.write_text(json.dumps(results, indent=2, default=str))
            print(f"Results saved to {args.output}")

    elif args.command == "interconnect":
        print(f"Comparing interconnects ({args.nodes} nodes, {args.data_mb} MB)...")
        results = compare_interconnects(
            data_size_mb=args.data_mb,
            num_nodes=args.nodes,
        )
        print(json.dumps(results, indent=2))

    elif args.command == "memory":
        print("Comparing memory technologies...")
        results = compare_memory_technologies()
        print(json.dumps(results, indent=2))

    elif args.command == "accelerator":
        print("Comparing accelerators...")
        manager = AcceleratorManager()
        results = manager.compare_accelerators()
        print(json.dumps(results, indent=2))

    elif args.command == "insitu":
        print("Testing in-situ analysis...")
        import numpy as np
        data = np.random.rand(100, 100, 100)
        analyzer = InSituAnalyzer()
        results = analyzer.analyze_pipeline(data)
        for name, result in results.items():
            print(f"  {name}: {result.compression_ratio:.1f}x compression")

    elif args.command == "fault":
        print("Estimating fault tolerance savings...")
        results = estimate_fault_tolerance_savings(
            job_runtime_hours=24,
            mtbf_hours=12,
            checkpoint_interval_minutes=5,
            recovery_time_minutes=2,
        )
        print(json.dumps(results, indent=2))

    elif args.command == "submit":
        print(f"Submitting job ({args.nodes} nodes, {args.gpus} GPUs, {args.tpus} TPUs)...")
        system = create_nexgen_system(
            num_nodes=args.nodes,
            num_gpus=args.gpus,
            num_tpus=args.tpus,
        )
        job = HPCJob(
            name="cli_job",
            num_nodes=args.nodes,
            estimated_runtime_s=3600,
            estimated_energy_kwh=100,
            gpu_per_node=args.gpus,
            tpu_per_node=args.tpus,
            data_size_gb=args.data_gb,
            arithmetic_intensity=args.intensity,
        )
        orchestrator = NextGenOrchestrator(system)
        result = orchestrator.submit_job(job)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
