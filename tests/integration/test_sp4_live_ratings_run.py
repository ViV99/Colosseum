"""Live ratings in a real training run (SP4 T2.2, spec block 2): ratings.json gets the bt section (the RandomBot
anchor at 0, snapshot buckets of the trainable agent), the ratings records get the compact bt summary, the console
shows `bt <rating> ± <ci>`."""
from __future__ import annotations

from cli_runner import run_train
from game_helpers import SP3_TTT_TINY


def test_a_run_publishes_live_ratings(tmp_path):
    run = run_train(SP3_TTT_TINY, tmp_path, name="live", tiny=False,
                    overrides={"ratings.refit_interval_sec": "1", "training.total_timesteps": "6000"})
    assert run.returncode == 0, run.stderr[-3000:]
    section = run.ratings()["bt"]["2p"]
    rows = section["player"]
    by_entity = {r["entity"]: r for r in rows}
    assert by_entity["random"]["anchor"] and by_entity["random"]["rating"] == 0.0
    assert section["@anchors"]["player/0"]["rule"] == "random_bot"
    buckets = sorted(r["version"] for r in rows if r["lineage"] == "main")
    assert buckets and buckets[0] == 0 and all(v % 20 == 0 for v in buckets)       # checkpoint.interval 20
    assert all(r["ci_low"] <= r["rating"] <= r["ci_high"] for r in rows)
    records = [r for r in run.records("ratings") if "bt" in r]
    assert records and records[-1]["bt"]["main"]["2p"]["latest"]["entity"] == f"main@v{buckets[-1]}"
    assert "| bt " in run.stderr and "Live ratings: Bradley-Terry refit every 1 s" in run.log("main")


def test_the_final_refit_covers_every_journaled_match(tmp_path):
    """Without a periodic refit (a long interval) ratings.json comes from the final refit, which runs after the
    journal writer flushed: its games are the journal's, match for match. A fresh run's table is empty when the
    thread starts, so the final refit is its only one (no ratings record before the last carries bt). A resume
    with the same long interval fits the parent's journal at once (T4.1), and its final refit still covers every
    match journaled after that first refit."""
    from colosseum.arena.matchlog import RUN_SOURCES, MatchLogReader

    def journal_games(run):
        with MatchLogReader(run.root / "matches.db") as reader:
            return reader.comparison_table(RUN_SOURCES).entity_games("2p", RUN_SOURCES)

    run = run_train(SP3_TTT_TINY, tmp_path, name="final", tiny=False,
                    overrides={"ratings.refit_interval_sec": "3600", "training.total_timesteps": "3000"})
    assert run.returncode == 0, run.stderr[-3000:]
    rows = run.ratings()["bt"]["2p"]["player"]
    journal = journal_games(run)
    assert journal and {r["entity"]: r["games"] for r in rows} == journal
    assert "no final refit" not in run.log("main")
    assert not any(r.get("bt") for r in run.records("ratings")[:-1])        # the final refit was the only one

    budget = run.ratings()["env_steps"] + 2000
    resumed = run_train(SP3_TTT_TINY, tmp_path, name="resumed", tiny=False,
                        overrides={"ratings.refit_interval_sec": "3600", "training.resume_from": str(run.root),
                                   "training.total_timesteps": str(budget)})
    assert resumed.returncode == 0, resumed.stderr[-3000:]
    early = [r for r in resumed.records("ratings")[:-1] if r.get("bt")]
    assert early, "the parent's journal was not fitted at the start"
    rows = resumed.ratings()["bt"]["2p"]["player"]
    journal = journal_games(resumed)
    assert sum(journal.values()) > sum(journal_games(run).values())     # matches were added after the first refit
    assert {r["entity"]: r["games"] for r in rows} == journal
