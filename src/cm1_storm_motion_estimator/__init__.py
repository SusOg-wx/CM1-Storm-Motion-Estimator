__all__ = [
    "__version__",
    "SimulationDataset",
    "CM1DataInspector",
    "compute_relative_vorticity",
    "estimate_motion",
    "CM1MainWindow",
]

__version__ = "0.1.0"

from .config import *
from .models import SimulationDataset
from .data.netcdf_inspector import CM1DataInspector
from .tracking.vorticity import compute_relative_vorticity
from .motion.estimate import estimate_motion
from .gui.main_window import CM1MainWindow
