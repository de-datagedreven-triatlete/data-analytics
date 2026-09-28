# utils/config.py
from dataclasses import dataclass

@dataclass
class AthleteConfig:
    max_hr: int = 187
    threshold_hr: int = 173
    ftp: int = 197

@dataclass
class PathConfig:
    fit_dir: str = "../fit_bestanden"
    output_dir: str = "../output"

# Standaard instanties
DEFAULT_ATHLETE = AthleteConfig()
DEFAULT_PATHS = PathConfig()