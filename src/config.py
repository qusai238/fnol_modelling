"""Project path configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

INCURRED_COL = "incurred"

# Large-loss threshold: claims above this are split off from the severity target
CAP = 50_000
# Target column names created below
TARGET_INCURRED = "y_incurred"   # floored at 0, no cap (evaluation only)
TARGET_CAPPED = "y_capped"       # floored at 0 and capped (severity model target)
TARGET_LARGE = "y_large"         # 1 if the claim is above the cap (classifier target)

# Out-of-time split: develop (tune and train) up to this date, test on later claims
DATE_COL = "date_of_loss"
DEV_END = "2013-12-31"
SPLIT_COL = "split"
SPLIT_TRAIN_VALID = "train_valid"      # tuning and final training
SPLIT_OOS = "oos"                      # out-of-time years 
CV_VALID_YEARS = (2011, 2012, 2013)    # validation years
TWEEDIE_EVAL_POWER = 1.5


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
INJURY_FLAGGED_COL = "inj_flagged_n"      
INJURY_UNCLEAR_COL = "tp_injury_unclear"
INJURY_UNKNOWN_COL = "tp_injury_nk"

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


ID_COLS = ["claim_number", "date_of_loss"]   # kept for splitting
TARGET_COLS = ["y_incurred", "y_capped", "y_large"]

CATEGORICAL_COLS = [
    "notifier", "location_of_incident", "weather_conditions",
    "vehicle_mobile", "main_driver", "ph_considered_tp_at_fault",
    "loss_season",
]

NUMERIC_COLS = [
    # timing and reporting
    "loss_month", "loss_day_of_week", "time_hour",
    "notification_period", "inception_to_loss",
    "loss_is_weekend",
    # cyclical timing
    "loss_hour_sin", "loss_hour_cos",
    "loss_month_sin", "loss_month_cos",
    "loss_day_of_week_sin", "loss_day_of_week_cos",
    "loss_day_of_year_sin", "loss_day_of_year_cos",
    # FNOL documents
    "vehicle_registration_present",
    "incident_details_present", "injury_details_present",
    # third-party type
    "tp_type_insd_pass_back", "tp_type_driver", "tp_type_pass_back",
    "tp_type_pass_front", "tp_type_bike", "tp_type_cyclist",
    "tp_type_pedestrian", "tp_type_other", "tp_type_nk",
    # injury
    "tp_injury_whiplash", "tp_injury_traumatic", "tp_injury_fatality",
    "tp_injury_unclear", "tp_injury_nk",
    # aggregates
    "tp_n_total", "inj_flagged_n", "inj_severity_ord", "inj_per_tp",
    "inj_known_share",
    # grouped regions
    "tp_reg_east", "tp_reg_london", "tp_reg_midlands", "tp_reg_north",
    "tp_reg_scotland", "tp_reg_south", "tp_reg_wales",
]

# Final feature and column lists for modeling
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
FINAL_COLS = ID_COLS + FEATURE_COLS + TARGET_COLS


# tuning variables 
SEED = 42
N_TRIALS = 100 
EARLY_STOPPING_ROUNDS = 100
MAX_TREES = 3000
FINAL_TREES_MULTIPLIER = 1.1


# MONTONIC_CONSTRAINTS 
MONOTONE_INCREASING_COLS = ["inj_severity_ord", "inj_flagged_n", "tp_n_total"]

SEVERITY_MIN_CHILD_WEIGHT_RANGE = (70.0, 7000.0)
LARGE_MIN_CHILD_WEIGHT_RANGE = (0.1, 5.0)

SEVERITY_DEPTH_RANGE = (2, 5)
LARGE_DEPTH_RANGE = (1, 3)         
TWEEDIE_POWER_RANGE = (1.3, 1.7)


# prediction columns 
PRED_CAPPED_COL = "pred_capped" # severity pred
PROB_LARGE_COL = "prob_large" # large loss prob prediction
PRED_TOTAL_COL = "pred_total" # prediction total 
PRED_BASELINE_COL = "pred_baseline" # severity baseline
PRED_BASELINE_TOTAL_COL = "pred_baseline_total"   # combined baseline: development mean of y_incurred
PREV_BASELINE_COL = "prev_baseline" # large_loss ratue used as baseline



# evaluation 
N_DECILES = 10
N_CALIBRATION_BINS = 5       
N_BOOTSTRAP = 500
RANDOM_SEED = 0

# Segments for actual-vs-expected tables 
SEGMENT_COLS = ["vehicle_mobile", "location_of_incident", "notifier",
                "weather_conditions", "injury_flagged"]

# Ways of turning the large claims into a flat excess p * mean/median/etc
EXCESS_METHODS = ["mean", "median", "trimmed_mean_10pct"]