import logging, warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")
logging.getLogger("wntr").setLevel(logging.ERROR)

from wdnv.config import Config
from wdnv.network import load_network
from wdnv.verifier import Model
from wdnv.incidents import Incident


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def net3(cfg):
    return Model.build(load_network("Net3", cfg), cfg)


@pytest.fixture(scope="session")
def ky6(cfg):
    return Model.build(load_network("ky6", cfg), cfg)


@pytest.fixture()
def leak109(cfg):
    # a sizeable burst on Net3 pipe 109 at 03:15
    return Incident("LEA001", "LEAK", 3 * 3600 + 900, {"pipe": "109", "area": 0.0038, "cd": cfg.leak_cd})
