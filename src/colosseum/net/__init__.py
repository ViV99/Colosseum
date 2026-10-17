"""The SP5 wire layer (no processes): array frames (``frame``), chunk / weights / checkpoint frames and the learner's
chunk checks (``chunks``), protobuf v1 and its converters (``proto``, ``messages``), identities, tokens, certificates
and node authentication (``security``), gRPC helpers (``grpc_util``) and same-machine connections (``local``).

Nothing in this package pickles data that crosses a machine, and nothing here starts a process or a thread.
"""
