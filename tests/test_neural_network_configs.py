from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.neural_network import get_architecture_configs


def test_returns_ten_configs():
    configs = get_architecture_configs()

    assert len(configs) == 10
    assert all("name" in cfg for cfg in configs)
    assert all("hidden_sizes" in cfg for cfg in configs)


def test_selu_activation_override():
    configs = get_architecture_configs(activation="selu")

    assert len(configs) == 10
    assert all(cfg["activation"] == "selu" for cfg in configs)
