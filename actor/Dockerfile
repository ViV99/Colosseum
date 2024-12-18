FROM python:3.12-slim

RUN apt-get update &&  \
    apt-get install -y --no-install-recommends curl build-essential  \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.4

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-ansi --no-interaction --no-root --without learner && rm -rf $POETRY_CACHE_DIR

COPY ./actor/ ./actor

ENV PORT=8000
EXPOSE $PORT
ENTRYPOINT ["poetry", "run", "python", "actor/main.py"]
