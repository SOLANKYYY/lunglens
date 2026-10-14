"""Generate repeatable artificial patterns, NOT patient scans or diagnostic labels."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
root = ROOT/'datasets/synthetic'
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
rows=[]
classes=['pattern_none','pattern_round','pattern_irregular']
for class_index,label in enumerate(classes):
    for i in range(60):
        split = 'train' if i<36 else 'val' if i<48 else 'test'
        # Separate artificial subjects, with independent noise and spatial variation.
        image = Image.new('L',(160,160),15); d=ImageDraw.Draw(image)
        shift=int(rng.integers(-4,5))
        d.ellipse((10+shift,24,151+shift,137),fill=int(rng.integers(115,135)),outline=195,width=3)
        d.ellipse((28+shift,39,71+shift,122),fill=30)
        d.ellipse((90+shift,39,134+shift,122),fill=30)
        d.ellipse((73+shift,111,91+shift,133),fill=200)
        cx=111+shift+int(rng.integers(-3,4));cy=77+int(rng.integers(-7,8))
        if class_index==1: d.ellipse((cx-6,cy-6,cx+6,cy+6),fill=190)
        if class_index==2:
            points=[(cx+int((15 if j%2==0 else 9)*np.cos(j*np.pi/6)),cy+int((15 if j%2==0 else 9)*np.sin(j*np.pi/6))) for j in range(12)]
            d.polygon(points,fill=210)
        pixels=np.asarray(image,dtype=float)+rng.normal(0,5,(160,160))
        image=Image.fromarray(np.clip(pixels,0,255).astype('uint8'))
        path=Path(split)/label/f'synthetic-{class_index}-{i:03}.png'
        (root/path).parent.mkdir(parents=True,exist_ok=True);image.save(root/path)
        rows.append({'path':path.as_posix(),'label':label,'patient_id':f'artificial-subject-{class_index}-{i:03}','split':split,'source':'synthetic'})
with (root/'manifest.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
(root/'DATA_CARD.md').write_text('''# Synthetic demonstration dataset\n180 generated grayscale PNG images; no patients and no real CT scans.\n108 train / 36 validation / 36 test. Three artificial pattern labels.\nGenerated with a fixed random seed. Original generated data supplied under CC0.\nShapes and noise are intentionally simple; high scores do not imply cancer detection.\nDo not rename these labels to medical classes or assign cancer stages.\n''')
preview=ROOT/'web/demo_samples';preview.mkdir(exist_ok=True)
import shutil
for label in classes: shutil.copy(root/f'test/{label}/synthetic-{classes.index(label)}-048.png',preview/f'{label}.png')
print('Generated 180 synthetic images and manifest.')
