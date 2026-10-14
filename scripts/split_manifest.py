"""Split a verified real-image manifest by patient, never by individual slice.
Input CSV columns: path,label,patient_id,source. Paths relative to the CSV directory.
Output is written beside the input to preserve relative paths.
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
path=Path(a.input)
with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
patients=defaultdict(list)
for r in rows:
    if not all(r.get(k,'').strip() for k in ['path','label','patient_id','source']): raise ValueError('Missing required fields; do not invent patient IDs.')
    if r['label'] not in ['normal','benign','malignant']: raise ValueError('Expected normal, benign or malignant labels.')
    patients[r['patient_id']].append(r)
by_class=defaultdict(list)
for pid,group in patients.items():
    if len({r['label'] for r in group})!=1: raise ValueError('Conflicting patient labels: '+pid)
    by_class[group[0]['label']].append(pid)
if set(by_class)!=set(['normal','benign','malignant']):raise ValueError('All three classes required.')
rng=random.Random(a.seed)
for label,ids in by_class.items():
    if len(ids)<5: raise ValueError('Need at least five verified patients in each class for this demonstration split.')
    ids.sort();rng.shuffle(ids);n_val=max(1,round(.2*len(ids)));n_test=max(1,round(.2*len(ids)))
    for i,pid in enumerate(ids):
        split='val' if i<n_val else 'test' if i<n_val+n_test else 'train'
        for row in patients[pid]:row['split']=split
out=path.with_name('manifest.csv')
if out.resolve()==path.resolve():raise ValueError('Name input raw_manifest.csv to avoid overwriting.')
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','label','patient_id','source','split'],extrasaction='ignore');w.writeheader();w.writerows(rows)
print(out)
