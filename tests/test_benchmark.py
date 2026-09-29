"""Tests for HPC benchmark suite with real algorithm benchmarks."""

from nexgen.benchmark import HPCBenchmark, run_full_benchmark_suite
from nexgen.hybrid import create_nexgen_system


class TestHPCBenchmark:
    def test_benchmark_initialization(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        benchmark = HPCBenchmark(system)
        assert benchmark.system.num_compute_nodes == 256

    def test_run_all(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        benchmark = HPCBenchmark(system)
        results = benchmark.run_all()
        assert len(results) == 7

    def test_benchmark_interconnect(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        benchmark = HPCBenchmark(system)
        result = benchmark.benchmark_interconnect()
        assert result.name == "interconnect"
        assert "technologies" in result.metrics
        assert "routing" in result.metrics

    def test_benchmark_memory(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        benchmark = HPCBenchmark(system)
        result = benchmark.benchmark_memory()
        assert result.name == "memory"
        assert "technologies" in result.metrics
        assert "placement" in result.metrics

    def test_benchmark_accelerator(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        benchmark = HPCBenchmark(system)
        result = benchmark.benchmark_accelerator()
        assert result.name == "accelerator"
        assert "accelerators" in result.metrics
        assert "offloading" in result.metrics

    def test_benchmark_insitu(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        benchmark = HPCBenchmark(system)
        result = benchmark.benchmark_insitu()
        assert result.name == "insitu"
        assert "methods" in result.metrics

    def test_benchmark_energy(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        benchmark = HPCBenchmark(system)
        result = benchmark.benchmark_energy()
        assert result.name == "energy"
        assert "scheduler" in result.metrics
        assert "dvfs" in result.metrics

    def test_benchmark_fault_tolerance(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        benchmark = HPCBenchmark(system)
        result = benchmark.benchmark_fault_tolerance()
        assert result.name == "fault_tolerance"
        assert "savings" in result.metrics

    def test_benchmark_scalability(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        benchmark = HPCBenchmark(system)
        result = benchmark.benchmark_scalability()
        assert result.name == "scalability"
        assert "scaling" in result.metrics


class TestRunFullBenchmarkSuite:
    def test_run_suite(self):
        system = create_nexgen_system(num_nodes=256, num_gpus=64, num_tpus=16)
        report = run_full_benchmark_suite(system)
        assert "num_benchmarks" in report
        assert "benchmarks" in report
        assert report["num_benchmarks"] == 7
