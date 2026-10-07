"""Enables ``python -m colosseum ...`` (used by the Docker/K8s entrypoints)."""

from colosseum.cli import main

if __name__ == "__main__":
    main()
