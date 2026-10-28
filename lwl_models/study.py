"""Stimulus design and the established reference plotting geometry."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("apple", "banana", "bathtub", "carrots", "comb", "peas", "potty", "toothbrush")
PAIRS = {
    "apple": {"Related": "banana", "Unrelated": "bathtub"},
    "banana": {"Related": "apple", "Unrelated": "toothbrush"},
    "bathtub": {"Related": "potty", "Unrelated": "apple"},
    "carrots": {"Related": "peas", "Unrelated": "potty"},
    "comb": {"Related": "toothbrush", "Unrelated": "peas"},
    "peas": {"Related": "carrots", "Unrelated": "comb"},
    "potty": {"Related": "bathtub", "Unrelated": "carrots"},
    "toothbrush": {"Related": "comb", "Unrelated": "banana"},
}
STIM_BOX = (110, 610, 290, 790)
AOI_BOX = (0, 620, 190, 810)


def validate_pairs(csv_path):
    """Read only design columns, if local gaze data are available."""
    import pandas as pd

    frame = pd.read_csv(csv_path, usecols=["TargetImage", "Condition", "DistracterImage"])
    # read_csv preserves file column order rather than usecols order.
    observed = set(frame[["TargetImage", "Condition", "DistracterImage"]]
                   .drop_duplicates().itertuples(index=False, name=None))
    expected = {(target, condition, distractor)
                for target, pairs in PAIRS.items() for condition, distractor in pairs.items()}
    if observed != expected:
        raise ValueError("Local trial pairings differ from lwl_models.study.PAIRS.")
    return len(observed)
