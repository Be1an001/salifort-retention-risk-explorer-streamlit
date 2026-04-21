from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NavigatorRegistryError(RuntimeError):
    """Base error for navigator registry loading and querying."""


class NavigatorRegistryNotFoundError(NavigatorRegistryError):
    """Raised when a required registry file does not exist."""


class NavigatorRegistryValidationError(NavigatorRegistryError):
    """Raised when a registry file exists but fails validation."""


@dataclass(frozen=True)
class NavigatorRegistryBundle:
    source_registry: dict[str, Any]
    truth_registry: dict[str, Any]
    drift_register: dict[str, Any]
    pace_phase_map: dict[str, Any]
    glossary: dict[str, Any]
    registry_paths: dict[str, Path]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_registry": self.source_registry,
            "truth_registry": self.truth_registry,
            "drift_register": self.drift_register,
            "pace_phase_map": self.pace_phase_map,
            "glossary": self.glossary,
            "registry_paths": {key: str(value) for key, value in self.registry_paths.items()},
        }
