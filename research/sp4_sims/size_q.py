import sqlite3,time,os
c=sqlite3.connect("m.db")
t=time.time(); c.execute("create index if not exists ix_seat_match on seat(match_id)"); print("index build",time.time()-t,flush=True)
a=c.execute("select agent from seat limit 1").fetchone()[0]
t=time.time()
r=c.execute("select s2.agent,count(*),avg(s1.score) from seat s1 join seat s2 on s1.match_id=s2.match_id and s1.seat!=s2.seat where s1.agent=? group by s2.agent",(a,)).fetchall()
print(a,len(r),"per-agent vs opponents",time.time()-t,flush=True)
t=time.time()
r=c.execute("select s1.agent,s2.agent,count(*),sum(s1.score>s2.score) from seat s1 join seat s2 on s1.match_id=s2.match_id and s1.seat=0 and s2.seat=1 group by 1,2").fetchall()
print("full pair matrix",len(r),time.time()-t,flush=True)
print(os.path.getsize("m.db")/1e6)
