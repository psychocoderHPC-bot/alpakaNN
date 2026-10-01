#!/usr/bin/env python3
"""Render truthful presentation PNGs from heat-closure CSV exports."""
import argparse, csv, json
from pathlib import Path
import numpy as np
try:
 import matplotlib; matplotlib.use('Agg')
 import matplotlib.pyplot as plt
except ImportError as e: raise SystemExit('matplotlib is required: python3 -m pip install matplotlib') from e

def load_run(path):
 p=Path(path); rows=list(csv.DictReader((p/'manifest.csv').open()))
 if not rows: raise ValueError(f'{p}: empty manifest')
 out=[]
 for row in rows:
  a=np.genfromtxt(p/f"frame_{int(row['frame']):06d}.csv", delimiter=',', names=True)
  n=len(a); nx=len(np.unique(a['x'])); ny=len(np.unique(a['y']))
  if n!=nx*ny or nx<2 or ny<2: raise ValueError('snapshot is not a complete rectangular grid')
  x=np.unique(a['x']); y=np.unique(a['y']); z=np.asarray(a['u']).reshape(ny,nx)
  out.append((float(row['time']),x,y,z))
 return out

def setup_diagram(out):
 fig,ax=plt.subplots(figsize=(12.8,7.2)); ax.set(xlim=(0,1),ylim=(0,1),xlabel='x',ylabel='y',title='Heat-closure setup (specified geometry, not a simulation result)'); ax.axvspan(0,.008,color='red',label='Hot wall: u=1'); ax.axvspan(.992,1,color='blue',label='Cold wall: u=0'); ax.plot([0,1],[0,0],color='black',lw=5,label='Insulated top/bottom'); ax.plot([0,1],[1,1],color='black',lw=5); ax.add_patch(plt.Circle((.35,.5),.12,color='#355c7d',label='Low-transport inclusion')); ax.add_patch(plt.Rectangle((.45,.45),.20,.10,color='#f6bd60',label='High-transport segment')); ax.legend(loc='upper right'); ax.set_aspect('equal'); fig.savefig(out/'setup_diagram.png',dpi=100); plt.close(fig)

def render(ref, out, other=None, width=1920, height=1080):
 out.mkdir(parents=True,exist_ok=True); plt.rcParams.update({'font.size':15})
 setup_diagram(out)
 for i,(t,x,y,u) in enumerate(ref):
  fig,ax=plt.subplots(1,3 if other else 1,figsize=(width/100,height/100), constrained_layout=True)
  axs=np.atleast_1d(ax); im=axs[0].pcolormesh(x,y,u,vmin=0,vmax=1,cmap='inferno',shading='nearest'); axs[0].set(title='Reference temperature',xlabel='x',ylabel='y',aspect='equal'); fig.colorbar(im,ax=axs[0],label='u [0,1]')
  if other:
   ot,ox,oy,v=other[min(range(len(other)),key=lambda k:abs(other[k][0]-t))]
   if v.shape!=u.shape or not np.allclose([ox[0],ox[-1],oy[0],oy[-1]],[x[0],x[-1],y[0],y[-1]]): raise ValueError('comparison grids differ')
   for a,z,title,cmap,lo,hi in [(axs[1],v,'Neural temperature','inferno',0,1),(axs[2],abs(v-u),'Absolute difference','magma',0,1)]:
    q=a.pcolormesh(x,y,z,vmin=lo,vmax=hi,cmap=cmap,shading='nearest'); a.set(title=title,xlabel='x',ylabel='y',aspect='equal'); fig.colorbar(q,ax=a)
  fig.suptitle(f'Heat closure simulation — t = {t:.6g} (physical time)'); fig.savefig(out/f'comparison_{i:06d}.png',dpi=100); plt.close(fig)
 return len(ref)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--reference',required=True); p.add_argument('--comparison'); p.add_argument('--output',required=True); a=p.parse_args(); n=render(load_run(a.reference),Path(a.output),load_run(a.comparison) if a.comparison else None); print(f'Rendered {n} frames; values sourced from CSV snapshots.')
if __name__=='__main__': main()
