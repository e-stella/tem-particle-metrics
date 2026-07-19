"""Auto-route decision: crowding score from a table + tier choice. No torch."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tem_particle_metrics.analysis import (  # noqa: E402
    AUTO_CROWDING_THRESHOLD, crowding_score, decide_tier,
)


def test_crowding_score():
    assert crowding_score(pd.DataFrame({"touching_group_id": [None, None, None]})) == 0.0
    assert crowding_score(pd.DataFrame({"touching_group_id": [1, 1, None, None]})) == 0.5
    assert crowding_score(pd.DataFrame({"touching_group_id": [3, 3, 3, None]})) == 0.75
    # degenerate: empty table / missing column -> 0 (never escalate on nothing)
    assert crowding_score(pd.DataFrame({"touching_group_id": []})) == 0.0
    assert crowding_score(pd.DataFrame({"x": [1, 2]})) == 0.0


def test_decide_tier():
    assert decide_tier(0.12) == 1                     # sparse frame -> tier-1
    assert decide_tier(0.80) == 2                     # aggregated -> tier-2
    assert decide_tier(AUTO_CROWDING_THRESHOLD) == 2  # at threshold -> escalate
    assert decide_tier(0.55, threshold=0.6) == 1      # tunable threshold
    assert decide_tier(0.0) == 1                       # empty frame stays tier-1


if __name__ == "__main__":
    test_crowding_score()
    test_decide_tier()
    print("ok")
