"""Project path configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Third-party (TP) type columns that describe a KNOWN type.
# "tp_type_nk" (not known) is handled separately.
# Dropped: tp_type_insd_pass_front and tp_type_pass_multi (always 0 in this data).
TP_TYPE_KNOWN_COLS = [
    "tp_type_insd_pass_back",
    "tp_type_driver",
    "tp_type_pass_back",
    "tp_type_pass_front",
    "tp_type_bike",
    "tp_type_cyclist",
    "tp_type_pedestrian",
    "tp_type_other",
]
TP_TYPE_UNKNOWN_COL = "tp_type_nk"

# Injury columns that are actually flagged by the notifier.
# tp_injury_unclear and tp_injury_nk are NOT counted as flagged injuries.
INJURY_WHIPLASH_COL = "tp_injury_whiplash"
INJURY_TRAUMATIC_COL = "tp_injury_traumatic"
INJURY_FATALITY_COL = "tp_injury_fatality"
INJURY_FLAGGED_COLS = [INJURY_WHIPLASH_COL, INJURY_TRAUMATIC_COL, INJURY_FATALITY_COL]

# Severity ordinal: higher = more serious (0 = no injury flagged)
INJURY_SEVERITY_ORDER = {
    "none": 0,
    "whiplash": 1,
    "traumatic": 2,
    "fatality": 3,
}

# TP region columns (one count column per region)
TP_REGION_NAMES = [
    "eastang", "eastmid", "london", "north", "northw", "outerldn",
    "scotland", "southe", "southw", "wales", "westmid", "yorkshire",
]
TP_REGION_PREFIX = "tp_region_"

# Region -> broader group (reduces sparsity: 12 columns become 7 groups)
TP_REGION_GROUP_MAP = {
    "london": "london",
    "outerldn": "london",
    "southe": "south",
    "southw": "south",
    "eastang": "east",
    "eastmid": "midlands",
    "westmid": "midlands",
    "north": "north",
    "northw": "north",
    "yorkshire": "north",
    "scotland": "scotland",
    "wales": "wales",
}