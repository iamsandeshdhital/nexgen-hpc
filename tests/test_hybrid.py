"""Tests for NextGen HPC orchestration with real algorithms."""

from nexgen.hybrid import (
    NextGenSystem,
    HPCJob,
    NextGenOrchestrator,
    create_nexgen_system,
    benchmark_nexgen_vs_traditional,
)


class TestNextGenSystem:
    def test_create_system(self):
        system = create_nexgen_system(num_nodes=1024, num_gpus=256, num_tpus=64)
        assert system.num_compute_nodes == 1024
        assert system.num_gpu_nodes == 256
        assert system.num_tpu_nodes == 64

    def test_total_nodes(self):
        system = create_nexgen_system(num_nodes=1024, num_gpus=256, num_tpus=64)
        assert system.total_nodes == 1344

    def test_total_memory_bandwidth(self):
        system = create_nexgen_system(num_nodes=1024, num_gpus=256, num_tpus=64)
        assert system.total_memory_bandwidth_tbps > 0

    def test_is_within_budget(self):
        system = create_nexgen_system(num_nodes=1024, num_gpus=256, num_tpus=64, power_budget_kw=10000)
        assert system.is_within_budget()


class TestNextGenOrchestrator:
    def test_submit_job(self):
        system = create_nexgen_system(num_nodes=1024, num_gpus=256, num_tpus=64)
        orchestrator = NextGenOrchestrator(system)

        job = HPCJob(
            name="test_job",
            num_nodes=128,
            estimated_runtime_s=3600,
            estimated_energy_kwh=100,
        )

        result = orchestrator.submit_job(job)
        assert result["status"] == "accepted"
        assert result["execution_time_s"] > 0

    def test_submit_job_rejected(self):
        system = create_nexgen_system(num_nodes=128, num_gpus=32, num_tpus=8)
        orchestrator = NextGenOrchestrator(system)

        job = HPCJob(
            name="test_job",
            num_nodes=256,
            estimated_runtime_s=3600,
            estimated_energy_kwh=100,
        )

        result = orchestrator.submit_job(job)
        assert result["status"] == "rejected"

    def test_optimize_job(self):
        system = create_nexgen_system(num_nodes=1024, num_gpus=256, num_tpus=64)
        orchestrator = NextGenOrchestrator(system)

        job = HPCJob(
            name="test_job",
            num_nodes=128,
            estimated_runtime_s=3600,
            estimated_energy_kwh=100,
            arithmetic_intensity=100,
            data_size_gb=100,
        )

        result = orchestrator.optimize_job(job)
        assert "recommended_accelerator" in result
        assert "speedup_vs_cpu" in result


class TestBenchmarkNextGenVsTraditional:
    def test_benchmark(self):
        result = benchmark_nexgen_vs_traditional(problem_size=10000)
        assert "traditional_time_s" in result
        assert "nexgen_time_s" in result
        assert "speedup" in result
