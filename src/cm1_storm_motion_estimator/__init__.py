__all__ = [
    "__version__",
    "SimulationDataset",
    "CM1DataInspector",
    "detect_updraft_regions",
    "detect_vorticity_regions",
    "estimate_motion",
    "CM1MainWindow",
]

__version__ = "0.1.0"

from .config import *
from .models import SimulationDataset
from .data.netcdf_inspector import CM1DataInspector
from .tracking.updraft import detect_updraft_regions
from .tracking.vorticity import detect_vorticity_regions
from .motion.estimate import estimate_motion
from .gui.main_window import CM1MainWindow
