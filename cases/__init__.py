# cases : one file per case study (cs1.py, cs2.py, ...), all with the same layout.
# load_case("cs1") returns the module cases/cs1.py and stops with a clear message if a
# required setting is still blank (None).
import importlib

REQUIRED = ["NAME", "DT", "U_LIMIT", "Y_LIMIT", "ACTION_REPEAT", "WARMUP_STEPS", "HOLD_STEPS",
            "START_SETPOINT", "TEST_SETPOINT", "TRAIN_RANGE", "TRAIN_STEPS", "Process"]


def load_case(name):
    """'cs1' -> the module cases/cs1.py, checked for blanks."""
    try:
        case = importlib.import_module(f"cases.{name}")
    except ModuleNotFoundError:
        raise SystemExit(f"There is no case study '{name}'. Expected the file cases/{name}.py")
    blank = [setting for setting in REQUIRED if getattr(case, setting, None) is None]
    if blank:
        raise SystemExit(f"cases/{name}.py is not filled in yet. Still blank: {', '.join(blank)}")
    case.ID = name          # remember the short name ("cs1"): used for runs/cs1 and results/cs1_*
    return case
