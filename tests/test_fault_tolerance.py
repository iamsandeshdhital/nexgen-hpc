"""Tests for fault tolerance with real algorithms."""

from nexgen.fault_tolerance import (
    FaultToleranceManager,
    Checkpoint,
    FailureEvent,
    estimate_fault_tolerance_savings,
    benchmark_fault_tolerance,
)


class TestCheckpoint:
    def test_create(self):
        cp = Checkpoint(
            job_id="test",
            timestamp=100.0,
            iteration=10,
            state_size_mb=100,
        )
        assert cp.job_id == "test"
        assert cp.iteration == 10


class TestFaultToleranceManager:
    def test_create_checkpoint(self):
        manager = FaultToleranceManager()
        cp = manager.create_checkpoint("job1", iteration=10, state_size_mb=100)
        assert cp.job_id == "job1"
        assert cp.iteration == 10

    def test_recover_job(self):
        manager = FaultToleranceManager()
        manager.create_checkpoint("job1", iteration=10, state_size_mb=100)
        result = manager.recover_job("job1", available_nodes=[1, 2, 3])
        assert result.success
        assert result.recovery_time_s > 0

    def test_recover_nonexistent_job(self):
        manager = FaultToleranceManager()
        result = manager.recover_job("nonexistent", available_nodes=[1, 2, 3])
        assert not result.success

    def test_predict_failure(self):
        manager = FaultToleranceManager()
        result = manager.predict_failure(1, {"temperature_c": 85, "memory_usage_percent": 95})
        assert result["risk_score"] > 0
        assert result["risk_level"] in ["low", "medium", "high"]

    def test_get_failure_statistics(self):
        manager = FaultToleranceManager()
        stats = manager.get_failure_statistics()
        assert stats["total_failures"] == 0

    def test_optimize_checkpoint_interval(self):
        manager = FaultToleranceManager()
        result = manager.optimize_checkpoint_interval(
            job_runtime_s=86400,
            mtbf_s=43200,
            checkpoint_time_s=60,
        )
        assert "optimal_interval_s" in result
        assert "overhead_percent" in result


class TestEstimateFaultToleranceSavings:
    def test_savings(self):
        result = estimate_fault_tolerance_savings(
            job_runtime_hours=24,
            mtbf_hours=12,
            checkpoint_interval_minutes=5,
            recovery_time_minutes=2,
        )
        assert result["time_saved_hours"] > 0
        assert result["time_saved_percent"] > 0


class TestBenchmarkFaultTolerance:
    def test_benchmark(self):
        result = benchmark_fault_tolerance(
            job_runtime_hours=24,
            mtbf_hours=12,
        )
        assert "recovery_result" in result
        assert "optimization" in result
