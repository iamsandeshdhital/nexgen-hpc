"""Tests for photonic interconnect with real algorithms."""

from nexgen.interconnect import (
    INFINIBAND_HDR,
    PHOTONIC_SI_PH,
    PHOTONIC_WDM,
    CPO_SWITCH,
    NetworkTopology,
    compare_interconnects,
    model_photonic_link,
    create_fat_tree_topology,
    benchmark_routing_algorithms,
)


class TestLinkSpecs:
    def test_infiniband_hdr(self):
        assert INFINIBAND_HDR.bandwidth_gbps == 200
        assert INFINIBAND_HDR.latency_us == 2.0

    def test_photonic_siph(self):
        assert PHOTONIC_SI_PH.bandwidth_gbps == 1600
        assert PHOTONIC_SI_PH.latency_us == 0.5

    def test_photonic_wdm(self):
        assert PHOTONIC_WDM.bandwidth_gbps == 12800
        assert PHOTONIC_WDM.latency_us == 0.3

    def test_cpo_switch(self):
        assert CPO_SWITCH.bandwidth_gbps == 51200
        assert CPO_SWITCH.latency_us == 0.1


class TestNetworkTopology:
    def test_add_link(self):
        topo = NetworkTopology(num_nodes=4)
        topo.add_link(0, 1, 1000, 1.0)
        assert 1 in topo.adjacency[0]
        assert 0 in topo.adjacency[1]

    def test_remove_link(self):
        topo = NetworkTopology(num_nodes=4)
        topo.add_link(0, 1, 1000, 1.0)
        topo.remove_link(0, 1)
        assert 1 not in topo.adjacency[0]

    def test_shortest_path(self):
        topo = NetworkTopology(num_nodes=4)
        topo.add_link(0, 1, 1000, 1.0)
        topo.add_link(1, 2, 1000, 1.0)
        topo.add_link(2, 3, 1000, 1.0)
        path = topo.shortest_path(0, 3)
        assert path == [0, 1, 2, 3]

    def test_all_pairs_shortest_paths(self):
        topo = NetworkTopology(num_nodes=4)
        topo.add_link(0, 1, 1000, 1.0)
        topo.add_link(1, 2, 1000, 1.0)
        topo.add_link(2, 3, 1000, 1.0)
        paths = topo.all_pairs_shortest_paths()
        assert (0, 3) in paths
        assert paths[(0, 3)] == [0, 1, 2, 3]

    def test_optimize_topology(self):
        topo = NetworkTopology(num_nodes=16)
        result = topo.optimize_topology(
            required_bandwidth_gbps=1000,
            max_latency_us=10.0,
            link_specs=PHOTONIC_WDM,
        )
        assert "num_links" in result
        assert "total_cost_w" in result

    def test_wavelength_assignment(self):
        topo = NetworkTopology(num_nodes=4)
        topo.add_link(0, 1, 1000, 1.0)
        topo.add_link(1, 2, 1000, 1.0)
        topo.add_link(2, 3, 1000, 1.0)
        coloring = topo.wavelength_assignment(num_wavelengths=8)
        assert len(coloring) > 0

    def test_power_optimization(self):
        topo = NetworkTopology(num_nodes=4)
        topo.add_link(0, 1, 1000, 1.0)
        topo.add_link(1, 2, 1000, 1.0)
        topo.add_link(2, 3, 1000, 1.0)
        traffic = np.zeros((4, 4))
        traffic[0, 3] = 100
        result = topo.power_optimization(traffic)
        assert "total_link_loads" in result

    def test_failure_recovery(self):
        topo = NetworkTopology(num_nodes=4)
        topo.add_link(0, 1, 1000, 1.0)
        topo.add_link(1, 2, 1000, 1.0)
        topo.add_link(2, 3, 1000, 1.0)
        topo.add_link(0, 3, 1000, 1.0)  # Redundant link
        result = topo.failure_recovery((0, 1))
        assert result["recovered"]


class TestCreateFatTreeTopology:
    def test_create(self):
        topo = create_fat_tree_topology(k=8, link_specs=PHOTONIC_WDM)
        assert topo.num_nodes > 0
        assert len(topo.link_capacities) > 0


class TestBenchmarkRoutingAlgorithms:
    def test_benchmark(self):
        results = benchmark_routing_algorithms(num_nodes=64)
        assert "shortest_path_time_s" in results
        assert "wavelength_assignment_time_s" in results


class TestCompareInterconnects:
    def test_compare(self):
        results = compare_interconnects(data_size_mb=1000, num_nodes=1024)
        assert "InfiniBand HDR" in results
        assert "Photonic WDM" in results
        assert "CPO Switch" in results


class TestModelPhotonicLink:
    def test_model(self):
        result = model_photonic_link(data_size_gb=100, distance_m=100)
        assert result["total_bandwidth_tbps"] > 0
        assert result["total_time_us"] > 0
        assert result["total_energy_j"] > 0


import numpy as np
