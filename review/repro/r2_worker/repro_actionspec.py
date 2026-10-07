import sys; sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
import numpy as np, gymnasium
from colosseum.core.action_spec import ActionSpec
nvec = np.array([2,3,4,5,6,7,8,9,10,11,12,13])
spec = ActionSpec.from_space(gymnasium.spaces.MultiDiscrete(nvec))
print("component order:", [c.name for c in spec.components])
print("num_categories per flat slot:", [c.num_categories for c in spec.components])
flat = np.arange(12, dtype=np.float32)  # flat slot i holds value i
print("decode(flat=[0..11]) ->", spec.decode(flat).tolist(), "(env expects dim i to come from its own head)")
# natural-order env mask: dim 2 has 4 options -> mask segment length 4 at natural position
mask = np.concatenate([np.ones(n, bool) for n in nvec]); mask[2+3:2+3+4] = [True, False, False, False]  # dim2: only action 0 legal
fm = spec.flatten_mask(mask)
c2 = [c for c in spec.components if c.name == "2"][0]
print("flatten_mask(ndarray natural order) segment used for comp '2':", fm[c2.mask_offset:c2.mask_offset+c2.mask_size].tolist(), "(should be [T,F,F,F])")
t = ActionSpec.from_space(gymnasium.spaces.Tuple([gymnasium.spaces.Discrete(i+2) for i in range(11)]))
print("tuple decode order ->", t.decode(np.arange(11, dtype=np.float32)))
