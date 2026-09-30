"""Helpers for loading the model files shipped with cenic_bench."""

from pathlib import Path

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant

PACKAGE_NAME = "cenic_bench"
PACKAGE_ROOT = Path(__file__).parent


def make_parser(plant: MultibodyPlant) -> Parser:
    """Create a parser for the plant that can resolve cenic_bench models."""
    parser = Parser(plant)
    package_map = parser.package_map()
    if not package_map.Contains(PACKAGE_NAME):
        package_map.Add(PACKAGE_NAME, str(PACKAGE_ROOT))
    return parser
