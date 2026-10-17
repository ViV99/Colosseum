#!/usr/bin/env bash
# Regenerate src/colosseum/net/proto/colosseum_v1_pb2.py and colosseum_v1_pb2_grpc.py from colosseum_v1.proto
# (grpcio-tools, in the dev extra). Run from anywhere; commit the generated files with the .proto.
# The test tests/unit/test_sp5_messages.py::test_the_committed_generated_code_matches_the_proto fails until it is run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-.venv/bin/python}"
env -u PYTHONPATH "$PY" -m grpc_tools.protoc -I src --python_out=src --grpc_python_out=src \
    src/colosseum/net/proto/colosseum_v1.proto
echo "generated: src/colosseum/net/proto/colosseum_v1_pb2.py, colosseum_v1_pb2_grpc.py"
