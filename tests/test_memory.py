"""Tests for near-memory computing with real algorithms."""

import numpy as np

from nexgen.memory import (
    DDR5,
    HBM3,
    HBM3E,
    HBM4,
    MemoryTier,
    MemoryHierarchy,
    DataObject,
    PIMKernel,
    compare_memory_technologies,
    benchmark_memory_tiers,
    optimize_data_placement,
)


class TestMemorySpecs:
    def test_ddr5(self):
        assert DDR5.bandwidth_gbps == 38.4
        assert DDR5.capacity_gb == 64

    def test_hbm3e(self):
        assert HBM3E.bandwidth_gbps == 4800
        assert HBM3E.capacity_gb == 36

    def test_hbm4(self):
        assert HBM4.bandwidth_gbps == 6400


class TestMemoryTier:
    def test_access_time(self):
        tier = MemoryTier(HBM3E, num_stacks=4, distance_ns=10)
        access_time = tier.access_time_ns(data_size_gb=1.0)
        assert access_time > 0

    def test_total_bandwidth(self):
        tier = MemoryTier(HBM3E, num_stacks=4, distance_ns=10)
        assert tier.total_bandwidth_gbps == 4800 * 4


class TestMemoryHierarchy:
    def test_place_data(self):
        tiers = [MemoryTier(HBM3E, num_stacks=4, distance_ns=10)]
        hierarchy = MemoryHierarchy(tiers)
        data = DataObject(
            name="test",
            size_gb=10,
            access_frequency=100,
            read_write_ratio=0.8,
            locality=0.9,
        )
        tier = hierarchy.place_data(data)
        assert tier == 0

    def test_migrate_data(self):
        tiers = [MemoryTier(HBM3E, num_stacks=4, distance_ns=10)]
        hierarchy = MemoryHierarchy(tiers)
        data = DataObject(
            name="test",
            size_gb=10,
            access_frequency=100,
            read_write_ratio=0.8,
            locality=0.9,
        )
        hierarchy.place_data(data)
        result = hierarchy.migrate_data("test", target_tier=0)
        assert result["success"]

    def test_optimize_placement(self):
        tiers = [MemoryTier(HBM3E, num_stacks=4, distance_ns=10)]
        hierarchy = MemoryHierarchy(tiers)
        data_objects = [
            DataObject(name="hot", size_gb=10, access_frequency=100, read_write_ratio=0.8, locality=0.9),
            DataObject(name="cold", size_gb=1000, access_frequency=1, read_write_ratio=0.2, locality=0.1),
        ]
        result = hierarchy.optimize_placement(data_objects)
        assert "placements" in result
        assert "total_cost" in result


class TestPIMKernel:
    def test_execute(self):
        tier = MemoryTier(HBM3E, num_stacks=4, distance_ns=10)
        kernel = PIMKernel(
            name="test",
            arithmetic_intensity=100,
            data_size_gb=10,
            compute_tflops=100,
        )
        result = kernel.execute(tier)
        assert result["compute_time_us"] > 0
        assert result["total_energy_j"] > 0


class TestCompareMemory:
    def test_compare(self):
        results = compare_memory_technologies()
        assert "DDR5" in results
        assert "HBM3e" in results
        assert "HBM4" in results


class TestBenchmarkMemoryTiers:
    def test_benchmark(self):
        results = benchmark_memory_tiers(data_size_gb=1.0)
        assert "DDR5" in results
        assert "HBM3e" in results
        assert "HBM4" in results


class TestOptimizeDataPlacement:
    def test_optimize(self):
        tiers = [MemoryTier(HBM3E, num_stacks=4, distance_ns=10)]
        hierarchy = MemoryHierarchy(tiers)
        data_objects = [
            DataObject(name="test", size_gb=10, access_frequency=100, read_write_ratio=0.8, locality=0.9),
        ]
        result = optimize_data_placement(data_objects, hierarchy)
        assert "placements" in result
