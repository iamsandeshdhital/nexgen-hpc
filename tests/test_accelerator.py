"""Tests for heterogeneous acceleration with real algorithms."""

from nexgen.accelerator import (
    NVIDIA_H100,
    NVIDIA_A100,
    AMD_MI300X,
    GOOGLE_TPU_V5P,
    AcceleratorManager,
    KernelProfile,
    benchmark_accelerator,
    optimize_offloading,
)


class TestAcceleratorSpecs:
    def test_h100(self):
        assert NVIDIA_H100.peak_tflops == 989
        assert NVIDIA_H100.memory_bandwidth_gbps == 3350

    def test_a100(self):
        assert NVIDIA_A100.peak_tflops == 312

    def test_mi300x(self):
        assert AMD_MI300X.peak_tflops == 1307
        assert AMD_MI300X.memory_bandwidth_gbps == 5300

    def test_tpu_v5p(self):
        assert GOOGLE_TPU_V5P.peak_tflops == 459
        assert GOOGLE_TPU_V5P.form_factor == "tpu"

    def test_energy_efficiency(self):
        assert NVIDIA_H100.energy_efficiency_gflops_per_w > 0


class TestKernelProfile:
    def test_gpu_suitable(self):
        kernel = KernelProfile(
            name="test",
            arithmetic_intensity=100,
            parallelism=10000,
            data_size_gb=10,
            control_flow_complexity=0.3,
        )
        assert kernel.is_gpu_suitable()

    def test_not_gpu_suitable(self):
        kernel = KernelProfile(
            name="test",
            arithmetic_intensity=1,
            parallelism=100,
            data_size_gb=10,
            control_flow_complexity=0.8,
        )
        assert not kernel.is_gpu_suitable()

    def test_tpu_suitable(self):
        kernel = KernelProfile(
            name="test",
            arithmetic_intensity=100,
            parallelism=100000,
            data_size_gb=10,
            control_flow_complexity=0.2,
        )
        assert kernel.is_tpu_suitable()

    def test_roofline_model(self):
        kernel = KernelProfile(
            name="test",
            arithmetic_intensity=100,
            parallelism=10000,
            data_size_gb=10,
        )
        result = kernel.roofline_model(NVIDIA_H100)
        assert "compute_roofline_tflops" in result
        assert "memory_roofline_tflops" in result
        assert "bottleneck" in result


class TestAcceleratorManager:
    def test_profile_kernel(self):
        manager = AcceleratorManager()
        kernel = manager.profile_kernel(
            name="test",
            arithmetic_intensity=100,
            parallelism=10000,
            data_size_gb=10,
        )
        assert kernel.name == "test"
        assert kernel.arithmetic_intensity == 100

    def test_recommend_accelerator(self):
        manager = AcceleratorManager()
        kernel = KernelProfile(
            name="test",
            arithmetic_intensity=100,
            parallelism=10000,
            data_size_gb=10,
            control_flow_complexity=0.3,
        )
        result = manager.recommend_accelerator(kernel)
        assert "recommended_accelerator" in result
        assert "speedup_vs_cpu" in result

    def test_load_balance(self):
        manager = AcceleratorManager()
        kernels = [
            KernelProfile(name="k1", arithmetic_intensity=100, parallelism=10000, data_size_gb=10),
            KernelProfile(name="k2", arithmetic_intensity=50, parallelism=5000, data_size_gb=5),
            KernelProfile(name="k3", arithmetic_intensity=10, parallelism=1000, data_size_gb=1),
        ]
        result = manager.load_balance(kernels, num_accelerators=2)
        assert "assignments" in result
        assert "makespan" in result

    def test_compare_accelerators(self):
        manager = AcceleratorManager()
        results = manager.compare_accelerators()
        assert "H100" in results
        assert "A100" in results
        assert "MI300X" in results
        assert "TPUv5" in results


class TestBenchmarkAccelerator:
    def test_benchmark(self):
        kernel = KernelProfile(
            name="test",
            arithmetic_intensity=100,
            parallelism=10000,
            data_size_gb=10,
        )
        result = benchmark_accelerator(kernel, NVIDIA_H100)
        assert result["total_time_ms"] > 0
        assert result["energy_j"] > 0
        assert result["bottleneck"] in ["compute", "memory"]


class TestOptimizeOffloading:
    def test_optimize(self):
        manager = AcceleratorManager()
        kernels = [
            KernelProfile(name="k1", arithmetic_intensity=100, parallelism=10000, data_size_gb=10),
            KernelProfile(name="k2", arithmetic_intensity=50, parallelism=5000, data_size_gb=5),
        ]
        result = optimize_offloading(kernels, manager)
        assert "decisions" in result
        assert "average_speedup" in result
