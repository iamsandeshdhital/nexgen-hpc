"""Tests for in-situ analysis with real algorithms."""

import numpy as np

from nexgen.insitu import (
    InSituAnalyzer,
    DataVolume,
    estimate_io_savings,
    benchmark_reduction_methods,
)


class TestDataVolume:
    def test_create(self):
        vol = DataVolume(
            name="test",
            size_gb=100,
            dimensions=(1000, 1000, 100),
            timesteps=10,
        )
        assert vol.total_size_gb == 1000
        assert vol.num_elements == 100000000


class TestInSituAnalyzer:
    def test_statistical_reduction(self):
        data = np.random.rand(100, 100, 100)
        analyzer = InSituAnalyzer()
        result = analyzer.statistical_reduction(data, "mean")
        assert result.compression_ratio > 1
        assert result.reduced_size_gb < result.original_size_gb

    def test_spatial_subsample(self):
        data = np.random.rand(100, 100, 100)
        analyzer = InSituAnalyzer()
        result = analyzer.spatial_subsample(data, factor=2)
        assert result.compression_ratio > 1

    def test_temporal_subsample(self):
        data = np.random.rand(100, 100, 100)
        analyzer = InSituAnalyzer()
        result = analyzer.temporal_subsample(data, factor=2)
        assert result.compression_ratio > 1

    def test_compressed_sensing(self):
        data = np.random.rand(100, 100)
        analyzer = InSituAnalyzer()
        result = analyzer.compressed_sensing(data, compression_ratio=0.1)
        assert result.compression_ratio > 1

    def test_feature_extraction(self):
        data = np.random.rand(100, 100)
        analyzer = InSituAnalyzer()
        result = analyzer.feature_extraction(data, num_features=10)
        assert result.compression_ratio > 1

    def test_adaptive_reduction(self):
        data = np.random.rand(50, 50, 50)
        analyzer = InSituAnalyzer()
        result = analyzer.adaptive_reduction(data, target_ratio=0.9)
        assert result.compression_ratio > 1

    def test_analyze_pipeline(self):
        data = np.random.rand(50, 50, 50)
        analyzer = InSituAnalyzer()
        results = analyzer.analyze_pipeline(data)
        assert len(results) > 0


class TestEstimateIOSavings:
    def test_savings(self):
        result = estimate_io_savings(
            data_size_gb=1000,
            reduction_ratio=0.9,
            io_bandwidth_gbps=10,
        )
        assert result["time_saved_s"] > 0
        assert result["time_saved_percent"] > 0


class TestBenchmarkReductionMethods:
    def test_benchmark(self):
        results = benchmark_reduction_methods(
            data_size_gb=0.001,
            dimensions=(100, 100, 100),
        )
        assert len(results) > 0
