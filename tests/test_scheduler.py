"""Tests for energy-aware scheduling with real algorithms."""

import numpy as np

from nexgen.scheduler import (
    NodeSpecs,
    Job,
    EnergyAwareScheduler,
    compare_scheduling_policies,
    benchmark_scheduler,
    optimize_frequency_sweep,
)


class TestNodeSpecs:
    def test_power_at_frequency(self):
        node = NodeSpecs(
            name="test",
            num_cores=128,
            memory_gb=512,
            tdp_w=500,
            base_freq_ghz=2.0,
            max_freq_ghz=3.5,
            idle_power_w=100,
        )
        power = node.power_at_frequency(freq_ghz=3.5)
        assert power > 0

    def test_performance_at_frequency(self):
        node = NodeSpecs(
            name="test",
            num_cores=128,
            memory_gb=512,
            tdp_w=500,
            base_freq_ghz=2.0,
            max_freq_ghz=3.5,
            idle_power_w=100,
        )
        perf = node.performance_at_frequency(freq_ghz=3.5)
        assert perf == 1.0

    def test_energy_efficiency_at_frequency(self):
        node = NodeSpecs(
            name="test",
            num_cores=128,
            memory_gb=512,
            tdp_w=500,
            base_freq_ghz=2.0,
            max_freq_ghz=3.5,
            idle_power_w=100,
        )
        efficiency = node.energy_efficiency_at_frequency(freq_ghz=3.0)
        assert efficiency > 0


class TestEnergyAwareScheduler:
    def test_schedule_job(self):
        nodes = [NodeSpecs(
            name="node",
            num_cores=128,
            memory_gb=512,
            tdp_w=500,
            base_freq_ghz=2.0,
            max_freq_ghz=3.5,
            idle_power_w=100,
        )] * 10

        scheduler = EnergyAwareScheduler(nodes, power_budget_kw=1000)
        job = Job(
            name="test_job",
            num_nodes=4,
            estimated_runtime_s=3600,
            estimated_energy_kwh=100,
            priority=5,
        )

        decision = scheduler.schedule_job(job)
        assert decision is not None
        assert decision.assigned_nodes is not None

    def test_backfill(self):
        nodes = [NodeSpecs(
            name="node",
            num_cores=128,
            memory_gb=512,
            tdp_w=500,
            base_freq_ghz=2.0,
            max_freq_ghz=3.5,
            idle_power_w=100,
        )] * 10

        scheduler = EnergyAwareScheduler(nodes, power_budget_kw=1000)
        job = Job(
            name="test_job",
            num_nodes=4,
            estimated_runtime_s=3600,
            estimated_energy_kwh=100,
            priority=5,
        )

        scheduler.schedule_job(job)
        backfill_job = Job(
            name="backfill",
            num_nodes=2,
            estimated_runtime_s=1800,
            estimated_energy_kwh=50,
            priority=3,
        )

        result = scheduler.backfill(backfill_job)
        assert result is not None

    def test_migrate_job(self):
        nodes = [NodeSpecs(
            name="node",
            num_cores=128,
            memory_gb=512,
            tdp_w=500,
            base_freq_ghz=2.0,
            max_freq_ghz=3.5,
            idle_power_w=100,
        )] * 10

        scheduler = EnergyAwareScheduler(nodes, power_budget_kw=1000)
        job = Job(
            name="test_job",
            num_nodes=4,
            estimated_runtime_s=3600,
            estimated_energy_kwh=100,
            priority=5,
        )

        decision = scheduler.schedule_job(job)
        new_decision = scheduler.migrate_job(decision, [5, 6, 7, 8])
        assert new_decision is not None

    def test_total_energy(self):
        nodes = [NodeSpecs(
            name="node",
            num_cores=128,
            memory_gb=512,
            tdp_w=500,
            base_freq_ghz=2.0,
            max_freq_ghz=3.5,
            idle_power_w=100,
        )] * 10

        scheduler = EnergyAwareScheduler(nodes, power_budget_kw=1000)
        job = Job(
            name="test_job",
            num_nodes=4,
            estimated_runtime_s=3600,
            estimated_energy_kwh=100,
            priority=5,
        )

        scheduler.schedule_job(job)
        assert scheduler.total_energy_kwh() > 0


class TestCompareSchedulingPolicies:
    def test_compare(self):
        nodes = [NodeSpecs(
            name="node",
            num_cores=128,
            memory_gb=512,
            tdp_w=500,
            base_freq_ghz=2.0,
            max_freq_ghz=3.5,
            idle_power_w=100,
        )] * 10

        jobs = [
            Job(name="job1", num_nodes=4, estimated_runtime_s=3600, estimated_energy_kwh=100, priority=5),
            Job(name="job2", num_nodes=2, estimated_runtime_s=1800, estimated_energy_kwh=50, priority=8),
        ]

        results = compare_scheduling_policies(jobs, nodes)
        assert "performance" in results
        assert "energy" in results
        assert "balanced" in results
        assert "carbon" in results


class TestBenchmarkScheduler:
    def test_benchmark(self):
        results = benchmark_scheduler(num_jobs=20, num_nodes=64)
        assert "performance" in results
        assert "energy" in results
        assert "balanced" in results
        assert "carbon" in results


class TestOptimizeFrequencySweep:
    def test_sweep(self):
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

        result = optimize_frequency_sweep(job, node)
        assert "optimal_frequency_ghz" in result
        assert "energy_savings_percent" in result
