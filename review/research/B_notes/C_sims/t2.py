import trueskill
from trueskill import Rating, rate, rate_1vs1, quality, TrueSkill, expose
print(trueskill.MU, trueskill.SIGMA, trueskill.BETA, trueskill.TAU, trueskill.DRAW_PROBABILITY)
a,b=Rating(),Rating()
a1,b1=rate_1vs1(a,b); print(a1,b1, expose(a1))
r = rate([(Rating(),),(Rating(),),(Rating(),),(Rating(),)], ranks=[0,1,1,3])
print(r)
env = TrueSkill(draw_probability=0.0); print(env.rate_1vs1(Rating(),Rating()))
print(quality_1 := trueskill.quality_1vs1(Rating(),Rating()))
