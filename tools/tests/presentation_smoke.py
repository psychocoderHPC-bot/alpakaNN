import importlib.util, tempfile, csv
from pathlib import Path
import numpy as np
p=Path(__file__).parents[1]/'render_heat_closure.py'; s=importlib.util.spec_from_file_location('render',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with tempfile.TemporaryDirectory() as d:
 root=Path(d); run=root/'run'; run.mkdir()
 with (run/'manifest.csv').open('w') as f:
  w=csv.writer(f); w.writerow(['frame','time','step','material','grid','beta']); w.writerow([0,0,0,'preset','2x2',.5])
 with (run/'frame_000000.csv').open('w') as f:
  w=csv.writer(f); w.writerow(['x','y','u']); w.writerows([[.25,.25,0],[.75,.25,.2],[.25,.75,.4],[.75,.75,1]])
 frames=m.load_run(run); assert frames[0][3].shape==(2,2); assert m.render(frames,root/'png')==1; assert (root/'png/comparison_000000.png').is_file()
