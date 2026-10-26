"""For the best-replayed TMNF-X track of each real circuit, download the fastest replay and compare the driven path length with the real lap length.
Usage: python path_lengths.py <tracks_found.json from find_real_circuits_tmx.py> <out.json>"""
import json, re, sys, requests, numpy as np, tempfile, os, time
sys.path.insert(0,"/home/user/racer-helper-extended-research/linesight/scripts")
from extract_driver_profile import _load_ghost
H="https://tmnf.exchange"; s=requests.Session()
real={"Monza":5.793,"Spa-Francorchamps":7.004,"Nürburgring":5.148,"Imola":4.909,"Silverstone":5.891,"Suzuka":5.807,"Laguna Seca":3.602,"Barcelona/Catalunya":4.675,"Brands Hatch":3.908,"Zandvoort":4.259,"Red Bull Ring":4.318,"Mugello":5.245,"Hockenheim":4.574,"Paul Ricard":5.842,"Le Mans":13.626,"Interlagos":4.309,"Monaco":3.337,"Bahrain":5.412,"COTA":5.513,"Bathurst":6.213,"Hungaroring":4.381,"Albert Park":5.278}
pat={"Monza":r"\bmonza\b","Spa-Francorchamps":r"francorchamps|\bspa\b(?!ce)","Nürburgring":r"n[uü]rburgring GP|F1 GP N","Imola":r"\bimola\b","Silverstone":r"silverstone","Suzuka":r"suzuka","Laguna Seca":r"laguna.?seca|WCG09-Laguna","Barcelona/Catalunya":r"barcelona|catalunya","Brands Hatch":r"brands.?hatch","Zandvoort":r"zandvoort","Red Bull Ring":r"red ?bull ring|spielberg|a1.?ring","Mugello":r"mugello","Hockenheim":r"hockenheim","Le Mans":r"\ble mans\b|la sarthe","Interlagos":r"interlagos","Monaco":r"\bmonaco\b","Bahrain":r"bahrain","COTA":r"\bcota\b","Hungaroring":r"hungaroring","Albert Park":r"albert park"}
tr=json.load(open(sys.argv[1]))
out=[]
for circuit,p in pat.items():
    c=sorted([x for x in tr if re.search(p,x["name"],re.I)],key=lambda x:-x["replays"])[:3]
    for x in c:
        if x["replays"]<2: continue
        d=s.get(H+"/api/replays",params=dict(trackId=x["id"],count=3,fields="ReplayId,ReplayTime,ReplayRespawns"),timeout=30).json()["Results"]
        d=[r for r in d if r["ReplayRespawns"]==0]
        if not d: continue
        r=d[0]; b=s.get(f"{H}/recordgbx/{r['ReplayId']}",timeout=60).content
        f=tempfile.NamedTemporaryFile(suffix=".Replay.Gbx",delete=False); f.write(b); f.close()
        try:
            g,_,rt,n=_load_ghost(f.name)
            P=np.array([[q.position.x,q.position.y,q.position.z] for q in g.records[:n+1]])
            L=float(np.linalg.norm(np.diff(P,axis=0),axis=1).sum())
            lap=len(g.cp_times)
            out.append(dict(circuit=circuit,track=x["name"],id=x["id"],replays=x["replays"],best_s=rt/1000,driven_km=L/1000,real_km=real[circuit],cps=lap))
        except SystemExit as e: print("skip",x["name"],e)
        finally: os.unlink(f.name)
        time.sleep(0.2)
json.dump(out,open(sys.argv[2],"w"),indent=1)
for o in out: print(f'{o["circuit"]:20s} {o["track"][:34]:34s} replays {o["replays"]:3d} best {o["best_s"]:6.1f}s driven {o["driven_km"]:5.2f} km real {o["real_km"]:5.2f} km ratio {o["driven_km"]/o["real_km"]:.2f}')
