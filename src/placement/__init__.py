"""VM placement strategy package for ThermoPlace."""

from src.placement.base import BasePlacementStrategy
from src.placement.first_fit import FirstFitPlacement

__all__ = ["BasePlacementStrategy", "FirstFitPlacement"]
