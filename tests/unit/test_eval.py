"""Tests for the evaluation module."""


from colosseum.eval import EvalMatrix, _wilson_ci, evaluate_agents
from colosseum.networks.actor_critic import ActorCriticNetwork
from examples.tic_tac_toe.env import TicTacToeEnv
from examples.tic_tac_toe.networks import TicTacToeEncoder, TicTacToePolicy, TicTacToeValue


def _make_net():
    return ActorCriticNetwork(TicTacToeEncoder(), TicTacToePolicy(), TicTacToeValue())


def test_wilson_ci_basic():
    lo, hi = _wilson_ci(50, 100)
    assert 0.4 < lo < 0.5
    assert 0.5 < hi < 0.6


def test_wilson_ci_extremes():
    lo, hi = _wilson_ci(0, 100)
    assert lo == 0.0 or lo < 0.02
    assert hi < 0.1

    lo, hi = _wilson_ci(100, 100)
    assert lo > 0.9
    assert hi == 1.0 or hi > 0.98


def test_wilson_ci_zero_total():
    lo, hi = _wilson_ci(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_evaluate_two_agents():
    net_a = _make_net()
    net_b = _make_net()

    agent_configs = {
        "a": {"state_dict": net_a.state_dict()},
        "b": {"state_dict": net_b.state_dict()},
    }

    matrix = evaluate_agents(
        agent_configs,
        env_fn=TicTacToeEnv,
        network_factory=_make_net,
        num_matches=20,
        num_envs=4,
    )

    assert len(matrix.results) == 2  # (a,b) and (b,a)

    r = matrix.get("a", "b")
    assert r is not None
    assert r.num_matches == 20
    assert r.wins_a + r.wins_b + r.draws == 20
    assert 0.0 <= r.win_rate_a <= 1.0
    assert r.ci_lower_a <= r.win_rate_a <= r.ci_upper_a

    # Reversed result should be consistent
    r2 = matrix.get("b", "a")
    assert r2 is not None
    assert r2.wins_a == r.wins_b
    assert r2.wins_b == r.wins_a


def test_eval_matrix_summary():
    matrix = EvalMatrix(agent_ids=["a", "b"])
    s = matrix.summary()
    assert isinstance(s, str)


def test_evaluate_three_agents():
    """Eval with 3 agents in a 2-player game: pairwise results for all 3 pairs."""
    net_a = _make_net()
    net_b = _make_net()
    net_c = _make_net()

    agent_configs = {
        "a": {"state_dict": net_a.state_dict()},
        "b": {"state_dict": net_b.state_dict()},
        "c": {"state_dict": net_c.state_dict()},
    }

    num_matches = 60
    matrix = evaluate_agents(
        agent_configs,
        env_fn=TicTacToeEnv,
        network_factory=_make_net,
        num_matches=num_matches,
        num_envs=4,
    )

    # 3 agents = 3 pairs, each stored both ways = 6 entries
    assert len(matrix.results) == 6

    # With slot rotation, each of the 3 agents actually plays. In a 2-player
    # game each match records exactly one pair, so the pairwise match counts
    # sum to the total number of matches, and every pair gets sampled.
    total_pair_matches = 0
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        r = matrix.get(a, b)
        assert r is not None, f"Missing result for ({a}, {b})"
        assert r.num_matches > 0, f"Pair ({a}, {b}) never played — rotation broken"
        assert r.wins_a + r.wins_b + r.draws == r.num_matches
        total_pair_matches += r.num_matches
    assert total_pair_matches == num_matches

    # Reversed results should be consistent
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        r = matrix.get(a, b)
        r2 = matrix.get(b, a)
        assert r2.wins_a == r.wins_b
        assert r2.wins_b == r.wins_a
