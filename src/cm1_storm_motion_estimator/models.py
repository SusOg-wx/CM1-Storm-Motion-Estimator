from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SimulationDataset:
    name: str
    file_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dimensions: Dict[str, int] = field(default_factory=dict)
    variables: Dict[str, List[str]] = field(default_factory=dict)
    x_coords: Optional[List[float]] = None
    y_coords: Optional[List[float]] = None
    z_coords: Optional[List[float]] = None
    time_coords: Optional[List[float]] = None
    grid_spacing_x: Optional[float] = None
    grid_spacing_y: Optional[float] = None

    @property
    def number_of_files(self) -> int:
        return len(self.file_paths)

    def add_file(self, path: str) -> None:
        self.file_paths.append(path)
