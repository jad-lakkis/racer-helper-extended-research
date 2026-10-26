"""Find TMNF-X tracks named after real circuits and count their replays (written for the circuit search in tracks/README.md)."""
import requests, time, json, sys
H="https://tmnf.exchange"
s=requests.Session()
names=["monza","spa","francorchamps","nurburgring","nordschleife","imola","silverstone","suzuka","laguna","barcelona","catalunya","brands hatch","zandvoort","red bull ring","spielberg","mugello","magione","vallelunga","hockenheim","paul ricard","le mans","interlagos","monaco","bahrain","yas marina","cota","austin","bathurst","donington","kyalami","highlands","trento","macau","albert park","baku","sepang","hungaroring","redbull"]
seen={}
for n in names:
    r=s.get(H+"/api/tracks",params=dict(name=n,count=100,fields="TrackId,TrackName,Authors"),timeout=30).json()
    for t in r["Results"]:
        seen.setdefault(t["TrackId"],(t["TrackName"],n,[a["User"]["Name"] for a in t["Authors"]]))
    time.sleep(0.15)
print(len(seen),"candidate tracks",file=sys.stderr)
rows=[]
for tid,(nm,q,au) in seen.items():
    cnt=0; after=None; best=None
    while True:
        p=dict(trackId=tid,count=1000,fields="ReplayId,User.UserId,ReplayTime,ReplayRespawns")
        if after: p["after"]=after
        d=s.get(H+"/api/replays",params=p,timeout=60).json()
        res=d["Results"]; cnt+=len(res)
        if res and best is None: best=min(x["ReplayTime"] for x in res)
        if not d["More"] or not res: break
        after=res[-1]["ReplayId"]
    users=len({x["User"]["UserId"] for x in res}) if cnt<=1000 else None
    rows.append(dict(id=tid,name=nm,query=q,authors=au,replays=cnt,wr_ms=best))
    time.sleep(0.1)
json.dump(rows,open(sys.argv[1],"w"))
