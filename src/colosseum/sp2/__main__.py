"""Enables ``python -m colosseum.sp2 ...`` (used by the Docker/K8s entrypoints)."""

from colosseum.sp2.cli import main

if __name__ == "__main__":
    main()
