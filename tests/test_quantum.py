"""Tests for quantum acceleration interface."""

from nexgen.quantum import (
    IBM_EAGLE,
    IBM_CONDOR,
    IONQ_HARMONY,
    QuantumKernel,
    QuantumAccelerator,
    compare_qpus,
)


class TestQuantumSpecs:
    def test_ibm_eagle(self):
        assert IBM_EAGLE.num_qubits == 127
        assert IBM_EAGLE.t1_us == 100

    def test_ibm_condor(self):
        assert IBM_CONDOR.num_qubits == 1121

    def test_ionq_harmony(self):
        assert IONQ_HARMONY.num_qubits == 32
        assert IONQ_HARMONY.connectivity == "all_to_all"

    def test_coherence_time(self):
        assert IBM_EAGLE.coherence_time_us == 100


class TestQuantumKernel:
    def test_execution_time(self):
        kernel = QuantumKernel(
            name="test",
            qpu=IBM_EAGLE,
            num_qubits=10,
            circuit_depth=100,
            shots=1024,
        )
        assert kernel.execution_time_ms > 0

    def test_success_probability(self):
        kernel = QuantumKernel(
            name="test",
            qpu=IBM_EAGLE,
            num_qubits=10,
            circuit_depth=100,
            shots=1024,
        )
        assert 0 < kernel.success_probability <= 1

    def test_is_feasible(self):
        kernel = QuantumKernel(
            name="test",
            qpu=IBM_EAGLE,
            num_qubits=10,
            circuit_depth=100,
        )
        assert kernel.is_feasible()


class TestQuantumAccelerator:
    def test_estimate_resources(self):
        accelerator = QuantumAccelerator(IBM_EAGLE)
        result = accelerator.estimate_resources(problem_size=100, problem_type="optimization")
        assert result["num_qubits"] > 0
        assert result["circuit_depth"] > 0

    def test_benchmark_vs_classical(self):
        accelerator = QuantumAccelerator(IBM_EAGLE)
        result = accelerator.benchmark_vs_classical(problem_size=100, problem_type="optimization")
        assert "classical_time_ms" in result
        assert "quantum_time_ms" in result


class TestCompareQPUs:
    def test_compare(self):
        results = compare_qpus(problem_size=100)
        assert "IBM Eagle" in results
        assert "IBM Condor" in results
        assert "IonQ Harmony" in results
