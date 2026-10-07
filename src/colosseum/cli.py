"""CLI entry point for Colosseum."""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """Colosseum — Distributed RL Training Framework."""
    pass


def _parse_overrides(overrides: tuple[str, ...]) -> dict:
    """Parse --set key=value pairs into a dict."""
    result = {}
    for ov in overrides:
        if "=" not in ov:
            raise click.BadParameter(
                f"Override must be in key=value format, got: {ov!r}"
            )
        key, value = ov.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
        result[key] = value
    return result


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--set", "overrides", multiple=True, help="Override config values (e.g., --set rollout.num_workers=8)")
def train(config: str, overrides: tuple[str, ...]) -> None:
    """Train an agent using the specified configuration."""
    from colosseum.launcher import run_training

    override_dict = _parse_overrides(overrides)
    run_training(config, overrides=override_dict if override_dict else None)


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option(
    "--data", "-d", required=True, type=click.Path(exists=True), help="Path to BC data (.pt file or directory)",
)
@click.option("--output", "-o", required=True, type=click.Path(), help="Path to save trained model weights (.pt)")
@click.option("--epochs", default=10, type=int, help="Number of BC training epochs")
@click.option("--batch-size", default=256, type=int, help="BC training batch size")
@click.option("--lr", default=1e-3, type=float, help="Learning rate for BC")
@click.option("--action-type", default="discrete", type=click.Choice(["discrete", "continuous"]))
def bc(
    config: str,
    data: str,
    output: str,
    epochs: int,
    batch_size: int,
    lr: float,
    action_type: str,
) -> None:
    """Train a policy via offline Behavioral Cloning."""
    import logging

    import torch

    from colosseum.bc.offline_bc import OfflineBCTrainer
    from colosseum.core.config import load_config
    from colosseum.core.registry import build_model

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(config)

    model = build_model(cfg)

    device = cfg.learner.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = OfflineBCTrainer(
        model=model,
        lr=lr,
        device=device,
        action_type=action_type,
    )
    trainer.load_data(data)
    metrics = trainer.train(num_epochs=epochs, batch_size=batch_size)

    # Save trained weights
    torch.save(model.state_dict(), output)
    click.echo(f"BC training complete. Loss={metrics['bc_loss']:.4f}")
    click.echo(f"Weights saved to {output}")


@main.command("eval")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--agents", "-a", required=True, multiple=True, help="Agent checkpoint paths (name:path.pt)")
@click.option("--num-matches", "-n", default=100, type=int, help="Matches per agent pair")
@click.option("--num-envs", default=8, type=int, help="Parallel environments for eval")
@click.option(
    "--deterministic", is_flag=True, default=False, help="Act greedily (distribution mode) instead of sampling",
)
def eval_cmd(config: str, agents: tuple[str, ...], num_matches: int, num_envs: int, deterministic: bool) -> None:
    """Evaluate agents/checkpoints against each other (no training)."""
    import logging

    import torch

    from colosseum.core.config import load_config
    from colosseum.core.registry import build_model, import_class
    from colosseum.eval import evaluate_agents

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(config)

    # Parse agent specs: "name:path.pt"
    agent_configs = {}
    for spec in agents:
        name, path = spec.split(":", 1)
        sd = torch.load(path, weights_only=True)
        agent_configs[name] = {"state_dict": sd}

    def env_fn():
        cls = import_class(cfg.env.env_class)
        return cls(**cfg.env.kwargs)

    def model_factory():
        return build_model(cfg)

    matrix = evaluate_agents(
        agent_configs, env_fn, model_factory,
        num_matches=num_matches, num_envs=num_envs,
        deterministic=deterministic,
    )

    click.echo("\n" + matrix.summary())


@main.command("validate")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
def validate_cmd(config: str) -> None:
    """Validate a config: build env + networks and run a dummy forward pass."""
    from colosseum.core.config import load_config
    from colosseum.core.registry import validate_config

    cfg = load_config(config)
    for aid in cfg.get_trainable_agent_ids():
        validate_config(cfg.get_agent_config(aid))
        click.echo(f"  OK: agent '{aid}'")
    click.echo("Config is valid.")


@main.command("run-learner")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--agent", "-a", default="agent_0", help="Trainable agent id this learner owns")
@click.option("--traj-port", default=50052, type=int, help="Port for this learner's TrajectoryService")
@click.option("--weight-store", required=True, help="WeightStore address host:port")
@click.option("--set", "overrides", multiple=True, help="Override config values (e.g., --set rollout.num_workers=8)")
def run_learner_cmd(config: str, agent: str, traj_port: int, weight_store: str, overrides: tuple[str, ...]) -> None:
    """Run one agent's learner as a gRPC service (distributed mode)."""
    from colosseum.distributed import run_distributed_learner

    override_dict = _parse_overrides(overrides)
    run_distributed_learner(
        config, agent, traj_port, weight_store,
        overrides=override_dict if override_dict else None,
    )


@main.command("run-workers")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--weight-store", required=True, help="WeightStore address host:port")
@click.option("--learner", "-l", "learners", required=True, multiple=True,
              help="Learner address per agent: agent_id=host:port (repeatable)")
@click.option("--set", "overrides", multiple=True, help="Override config values")
def run_workers_cmd(config: str, weight_store: str, learners: tuple[str, ...], overrides: tuple[str, ...]) -> None:
    """Run rollout workers feeding remote learners over gRPC (distributed mode)."""
    from colosseum.distributed import run_distributed_workers

    learner_addresses: dict[str, str] = {}
    for spec in learners:
        if "=" not in spec:
            raise click.BadParameter(f"--learner must be agent_id=host:port, got {spec!r}")
        aid, addr = spec.split("=", 1)
        learner_addresses[aid] = addr

    override_dict = _parse_overrides(overrides)
    run_distributed_workers(
        config, weight_store, learner_addresses,
        overrides=override_dict if override_dict else None,
    )


@main.command("serve-weight-store")
@click.option("--port", default=50051, type=int, help="gRPC port")
@click.option("--max-message-mb", default=64, type=int, help="Max gRPC message size in MiB")
def serve_weight_store_cmd(port: int, max_message_mb: int) -> None:
    """Start a gRPC weight store server."""
    import logging

    from colosseum.weight_store.grpc_store import serve_weight_store

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = serve_weight_store(port=port, max_message_mb=max_message_mb)
    click.echo(f"Weight store serving on port {port}. Press Ctrl+C to stop.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


@main.command("serve-trajectory")
@click.option("--port", default=50052, type=int, help="gRPC port")
@click.option("--max-message-mb", default=64, type=int, help="Max gRPC message size in MiB")
def serve_trajectory_cmd(port: int, max_message_mb: int) -> None:
    """Start a gRPC trajectory receiver server (for learner)."""
    import logging
    import queue

    from colosseum.transport.grpc_transport import serve_trajectory_receiver

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    chunk_queue = queue.Queue(maxsize=256)
    server = serve_trajectory_receiver(chunk_queue, port=port, max_message_mb=max_message_mb)
    click.echo(f"Trajectory receiver serving on port {port}. Press Ctrl+C to stop.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()
