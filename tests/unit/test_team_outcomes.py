"""resolve_outcome: default score = mean of seat returns, ranks from scores, checks (SP2 T1.4)."""
import math

import pytest

from colosseum.core.errors import EnvContractError
from colosseum.sp2.core.outcomes import pairwise_rank_score, resolve_outcome
from colosseum.sp2.envs.game import Outcome

TEAMS_2V1V1 = [[0, 1], [2], [3]]


def test_default_score_is_the_mean_of_seat_returns():
    rank, score = resolve_outcome(None, TEAMS_2V1V1, [3.0, 1.0, 2.0, 2.5])
    assert score == {0: 2.0, 1: 2.0, 2: 2.5}          # team 0: (3 + 1) / 2, not 4
    assert rank == {0: 2.0, 1: 2.0, 2: 1.0}           # ties share a rank; 1 + strictly better teams


def test_ranks_from_scores_with_ties():
    rank, _ = resolve_outcome(Outcome(team_score={0: 5.0, 1: 7.0, 2: 5.0, 3: 1.0}),
                              [[0], [1], [2], [3]], [0.0] * 4)
    assert rank == {0: 2.0, 1: 1.0, 2: 2.0, 3: 4.0}


def test_rank_only_keeps_the_mean_score():
    rank, score = resolve_outcome(Outcome(team_rank={0: 2, 1: 1, 2: 2.5}), TEAMS_2V1V1, [1.0, 0.0, 4.0, 9.0])
    assert rank == {0: 2.0, 1: 1.0, 2: 2.5}
    assert score == {0: 0.5, 1: 4.0, 2: 9.0}


def test_both_fields_are_taken_as_given():
    rank, score = resolve_outcome(Outcome(team_rank={0: 1, 1: 2}, team_score={0: -3.0, 1: 10.0}),
                                  [[0], [1]], [0.0, 0.0])
    assert rank == {0: 1.0, 1: 2.0} and score == {0: -3.0, 1: 10.0}


def test_empty_outcome_is_the_default():
    assert resolve_outcome(Outcome(), [[0], [1]], [1.0, 0.0]) == ({0: 1.0, 1: 2.0}, {0: 1.0, 1: 0.0})


def test_single_team_has_rank_one():
    assert resolve_outcome(None, [[0, 1]], [1.0, 3.0]) == ({0: 1.0}, {0: 2.0})


@pytest.mark.parametrize("outcome, message", [
    (Outcome(team_rank={0: 1, 1: 2}), r"team_rank keys \[0, 1\] must be exactly the layout's teams \[0, 1, 2\]"),
    (Outcome(team_score={0: 1.0, 1: 2.0, 2: 3.0, 3: 0.0}), "team_score keys"),
    (Outcome(team_score={0: 1.0, 1: math.nan, 2: 3.0}), r"team_score\[1\] must be finite"),
    (Outcome(team_rank={0: 1, 1: "first", 2: 3}), r"team_rank\[1\] must be a number"),
])
def test_bad_outcomes_raise_with_context(outcome, message):
    with pytest.raises(EnvContractError, match="worker 0, env 2: outcome") as info:
        resolve_outcome(outcome, TEAMS_2V1V1, [0.0] * 4, where="worker 0, env 2")
    assert info.match(message)


def test_pairwise_rank_score():
    assert pairwise_rank_score(1, 2) == 1.0
    assert pairwise_rank_score(2.5, 2.5) == 0.5
    assert pairwise_rank_score(3, 1) == 0.0
