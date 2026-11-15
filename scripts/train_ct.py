"""Train on the author's unmodified IQ-OTH/NCCD JPEGs. No synthetic training images.
Usage: python -m scripts.train_ct --source ../iqoth-source
"""
import argparse, csv, hashlib, json, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold
from backend.ct_model import features, kernel, CLASSES
from backend.common import evaluate

ROOT=Path(__file__).resolve().parents[1]

def train(source):
    files=[]; labels=[]; hashes=[]; phashes=[]; xs=[]
    for folder,label in [('Normal cases','normal'),('Bengin cases','benign'),('Malignant cases','malignant')]:
        for p in sorted((source/folder).glob('*.jpg')):
            data=p.read_bytes(); files.append(p); labels.append(CLASSES.index(label)); xs.append(features(data))
            with Image.open(p) as im:
                gray=im.convert('L')
                hashes.append(hashlib.sha256(gray.tobytes()).hexdigest())
                low=dctn(np.asarray(gray.resize((32,32)),dtype=float),norm='ortho')[:8,:8].reshape(-1)[1:]
                phashes.append(low>np.median(low))
    x=np.array(xs); y=np.array(labels); ph=np.array(phashes)
    parent=list(range(len(files)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    for i in range(len(files)):
        for j in range(i):
            if hashes[i]==hashes[j] or np.count_nonzero(ph[i]!=ph[j])<=4:
                parent[find(i)]=find(j)
    groups=np.array([find(i) for i in range(len(files))])
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(x,y,groups))
    test=folds[0][1]; val=folds[1][1]; tr=np.setdiff1d(np.arange(len(x)),np.r_[test,val])
    assert not (set(groups[tr]) & set(groups[test]) or set(groups[tr]) & set(groups[val]) or set(groups[val]) & set(groups[test]))
    target=np.eye(3)[y[tr]]
    best=None; trials=[]
    for gamma in [5.,20.,80.]:
        k=kernel(x[tr],x[tr],gamma); vk=kernel(x[val],x[tr],gamma)
        for regularization in [.01,.1,1.]:
            weights=np.linalg.solve(k+regularization*np.eye(len(tr)),target)
            metric=evaluate(y[val],vk@weights,CLASSES)
            trials.append({'gamma':gamma,'regularization':regularization,'validation_macro_f1':metric['macro_f1']})
            if best is None or metric['macro_f1']>best[0]: best=(metric['macro_f1'],gamma,regularization,weights)
    _,gamma,reg,weights=best
    out=ROOT/'models/ct'; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'weights.npz',reference=x[tr],alpha=weights)
    split={int(i):s for s,ids in [('train',tr),('validation',val),('test',test)] for i in ids}
    manifest=[]
    for i,p in enumerate(files):
        manifest.append({'path':p.relative_to(source).as_posix(),'label':CLASSES[y[i]],'split':split[i], 'similarity_group':int(groups[i]),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    limitation='Image-level split grouped by exact decoded pixels and perceptual hash distance <=4. Patient IDs are unavailable; patient overlap and near-duplicate leakage may remain. This is NOT patient-independent or clinical validation.'
    meta={'model_version':'iqoth-kernel-v1','architecture':'RBF kernel ridge classifier','framework':'NumPy','data_kind':'research','classes':CLASSES,'image_size':64,'gamma':gamma,'regularization':reg,'seed':42,'clinical_validation':False,'stage_supported':False,'source':'https://github.com/hamdalla93/The-IQ-OTHNCCD-lung-cancer-dataset','source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'dataset_doi':'10.17632/bhmdr45bh2.4','license':'CC BY 4.0','split_limitation':limitation,'weights_sha256':hashlib.sha256((out/'weights.npz').read_bytes()).hexdigest()}
    result={'images':len(files),'class_counts':{c:int(sum(y==i)) for i,c in enumerate(CLASSES)},'similarity_groups':len(set(groups)),'split_sizes':{s:len(ids) for s,ids in [('train',tr),('validation',val),('test',test)]},'selection':'Validation macro F1; held-out test evaluated once after selecting hyperparameters. Training partition only fits weights.','limitation':limitation,'trials':trials,'test':evaluate(y[test],kernel(x[test],x[tr],gamma)@weights,CLASSES)}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)); (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    public=ROOT/'web/real_samples'; public.mkdir(exist_ok=True)
    samples=[]
    # First two sorted held-out examples per label, never chosen by correctness.
    for i,c in enumerate(CLASSES):
        for n,idx in enumerate([j for j in test if y[j]==i][:2],1):
            name=f'{c}-{n}.jpg'; shutil.copyfile(files[idx],public/name)
            samples.append({'file':name,'label':c,'original_file':files[idx].relative_to(source).as_posix(),'sha256':manifest[idx]['sha256'],'split':'test'})
    (public/'samples.json').write_text(json.dumps(samples,indent=2))
    for name in ['metadata.json','evaluation.json']: shutil.copyfile(out/name,public/name)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    train(parser.parse_args().source.resolve())
