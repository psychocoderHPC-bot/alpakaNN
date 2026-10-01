#!/usr/bin/env python3
"""Assemble a 45 s captioned video from rendered CSV-derived PNGs; no fabricated metrics."""
import argparse, shutil, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def card(path, title, subtitle):
 im=Image.new('RGB',(1280,720),'#14243a'); d=ImageDraw.Draw(im)
 try: f=ImageFont.truetype('DejaVuSans.ttf',42); small=ImageFont.truetype('DejaVuSans.ttf',27)
 except OSError: f=small=ImageFont.load_default()
 d.text((70,230),title,font=f,fill='white'); d.text((70,320),subtitle,font=small,fill='#c9d7e8'); im.save(path)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--frames',required=True); p.add_argument('--output',required=True); p.add_argument('--sequence',default='presentation_sequence'); a=p.parse_args()
 src=sorted(Path(a.frames).glob('comparison_*.png'))
 if not src: raise SystemExit('No comparison_*.png frames; run render_heat_closure.py first.')
 if not shutil.which('ffmpeg') or not shutil.which('ffprobe'): raise SystemExit('ffmpeg and ffprobe are required (apt install ffmpeg).')
 seq=Path(a.sequence); seq.mkdir(parents=True,exist_ok=True)
 # 30 fps storyboard; input frame selection is a frame hold, not solution interpolation.
 cards=[('Neural material closure with alpakaNN','CSV-derived simulation imagery; accelerated playback'),('Analytical reference','Temperature data from exported physical-time snapshots'),('Matched comparison','Frame holds only; no interpolation of numerical fields'),('Learned material map','Coefficient arrays unavailable in this export; not shown'),('Host-only inference path','CPU solver and inference; device-resident integration is not implemented')]
 paths=[]
 for i,(title,sub) in enumerate(cards):
  q=seq/f'card_{i}.png'; card(q,title,sub); paths.append(q)
 # Make 1350 numbered frames, selecting real rendered frames for animation and explicitly labeled cards otherwise.
 for n in range(1350):
  sec=n//30
  if sec<5: im=Image.open(paths[0]).copy()
  elif sec<15: im=Image.open(src[min(len(src)-1,(sec-5)*len(src)//10)]).convert('RGB').resize((1280,720))
  elif sec<30: im=Image.open(src[min(len(src)-1,(sec-15)*len(src)//15)]).convert('RGB').resize((1280,720))
  else: im=Image.open(paths[3 if sec<40 else 4]).copy()
  im.save(seq/f'frame_{n:06d}.png')
 cmd=['ffmpeg','-y','-v','error','-framerate','30','-i',str(seq/'frame_%06d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',a.output]
 subprocess.run(cmd,check=True); probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,nb_frames','-show_entries','format=duration','-of','default=nw=1',a.output],check=True,text=True,capture_output=True); print(probe.stdout)
 subprocess.run(['ffmpeg','-v','error','-i',a.output,'-f','null','-'],check=True); print('Decode validation passed; duration is 45 s at 30 fps.')
if __name__=='__main__': main()
