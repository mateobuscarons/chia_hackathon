"""Tier 0: the pure helpers of the configuration build, no Ray, no ChampSim."""

from chia.simulators.champsim_config import check_config, config_name


def test_config_name_depends_on_the_configuration_only():
    first = {"L1D": {"sets": 64, "ways": 8}, "L2C": {"prefetcher": "spp_dev"}}
    same = {"L2C": {"prefetcher": "spp_dev"}, "L1D": {"ways": 8, "sets": 64}, "executable_name": "anything"}
    other = {"L1D": {"sets": 128, "ways": 8}, "L2C": {"prefetcher": "spp_dev"}}
    assert config_name(first) == config_name(same)
    assert config_name(first) != config_name(other)
    assert config_name(first).startswith("champsim_")


def test_check_config_accepts_a_buildable_configuration():
    config = {"executable_name": "x", "L1D": {"sets": 64, "ways": 12, "prefetcher": "va_ampm_lite"},
              "L2C": {"sets": 1024, "ways": 8, "replacement": "srrip", "mshr_size": 32},
              "LLC": {"sets": 2048, "ways": 16}, "physical_memory": {"data_rate": 4800}}
    assert check_config(config) == []


def test_check_config_names_each_problem():
    config = {"L1D": {"sets": 96, "ways": 0}, "LLC": {"prefetcher": "spp dev", "mshr_size": -1}}
    problems = check_config(config)
    assert any("L1D: sets" in problem for problem in problems)
    assert any("L1D: ways" in problem for problem in problems)
    assert any("LLC: prefetcher" in problem for problem in problems)
    assert any("LLC: mshr_size" in problem for problem in problems)
    assert len(problems) == 4
