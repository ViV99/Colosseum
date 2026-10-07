import torch
from colosseum.core.types import WeightPayload
from colosseum.weight_store.grpc_store import GRPCWeightStore, serve_weight_store
srv = serve_weight_store(port=51114)  # default max_message_mb=64 (same as CLI default)
st = GRPCWeightStore("localhost:51114")  # default 64
sd = {"w": torch.randn(4096, 4500)}  # 18.4M params = 70 MiB fp32
try:
    st.put("a", WeightPayload("a", 1, sd)); print("put ok")
except Exception as e:
    print("PutWeights failed:", type(e).__name__, e.code(), e.details()[:120])
st.close(); srv.stop(0)
