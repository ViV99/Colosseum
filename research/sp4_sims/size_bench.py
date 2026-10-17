import json, sqlite3, random, os, gzip, time, sys
random.seed(0)
N=1_000_000
agents=[f"agent_{i}@v{random.randint(1,400)}" for i in range(500)]
def rec(i):
    a,b=random.sample(agents,2)
    s0=random.random()
    return {"id":i,"t":1.7e9+i*0.5,"layout":"1v1","seed":random.getrandbits(32),"env_steps":random.randint(50,500),
            "seats":[{"agent":a,"hash":"%08x"%random.getrandbits(32),"role":"p","team":0,"score":s0,"rank":1 if s0>.5 else 2},
                     {"agent":b,"hash":"%08x"%random.getrandbits(32),"role":"p","team":1,"score":1-s0,"rank":2 if s0>.5 else 1}],
            "dur_s":round(random.random()*3,3),"worker":random.randint(0,63),"replay":None}
recs=[rec(i) for i in range(N)]
with open("m.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r,separators=(",",":"))+"\n")
print("jsonl MB",os.path.getsize("m.jsonl")/1e6)
with open("m.jsonl","rb") as f, gzip.open("m.jsonl.gz","wb") as g: g.write(f.read())
print("jsonl.gz MB",os.path.getsize("m.jsonl.gz")/1e6)
if os.path.exists("m.db"): os.remove("m.db")
c=sqlite3.connect("m.db")
c.execute("create table match(id integer primary key,t real,layout text,seed integer,steps integer,dur real,worker integer,replay text)")
c.execute("create table seat(match_id integer,seat integer,agent text,hash text,role text,team integer,score real,rank integer)")
c.execute("create index ix_seat_agent on seat(agent)")
t=time.time()
c.executemany("insert into match values(?,?,?,?,?,?,?,?)",((r["id"],r["t"],r["layout"],r["seed"],r["env_steps"],r["dur_s"],r["worker"],None) for r in recs))
c.executemany("insert into seat values(?,?,?,?,?,?,?,?)",((r["id"],k,s["agent"],s["hash"],s["role"],s["team"],s["score"],s["rank"]) for r in recs for k,s in enumerate(r["seats"])))
c.commit(); print("sqlite insert s",time.time()-t, "MB",os.path.getsize("m.db")/1e6)
t=time.time()
print(c.execute("select count(*) from seat s1 join seat s2 on s1.match_id=s2.match_id and s1.seat=0 and s2.seat=1 where s1.agent='agent_1@v5'").fetchall(), "query s",time.time()-t)
try:
    import pyarrow as pa, pyarrow.parquet as pq
    flat=[{"id":r["id"],"t":r["t"],"seed":r["seed"],"steps":r["env_steps"],"a0":r["seats"][0]["agent"],"a1":r["seats"][1]["agent"],"s0":r["seats"][0]["score"],"s1":r["seats"][1]["score"]} for r in recs]
    pq.write_table(pa.Table.from_pylist(flat),"m.parquet",compression="zstd")
    print("parquet MB",os.path.getsize("m.parquet")/1e6)
except Exception as e: print("no pyarrow",e)
