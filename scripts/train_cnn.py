"""Train one CNN using the frozen baseline manifest. Test only after checkpoint selection."""
import argparse,hashlib,json,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from backend.cnn_architecture import LungCNN
from backend.ct_model import features,CLASSES
from backend.common import evaluate
ROOT=Path(__file__).resolve().parents[1]
def train(source,epochs):
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    manifest=json.loads((ROOT/'models/ct/manifest.json').read_text())
    data={}
    for split in ['train','validation','test']:
        rows=[r for r in manifest if r['split']==split]
        xs=[];ys=[]
        for r in rows:
            raw=(source/r['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==r['sha256']
            xs.append(features(raw).astype('float32').reshape(1,64,64));ys.append(CLASSES.index(r['label']))
        data[split]=(torch.from_numpy(np.array(xs)),torch.tensor(ys))
    x,y=data['train'];vx,vy=data['validation'];tx,ty=data['test']
    out=ROOT/'models/cnn';out.mkdir(exist_ok=True)
    model=LungCNN();counts=torch.bincount(y,minlength=3).float()
    loss_fn=nn.CrossEntropyLoss(weight=len(y)/(3*counts));optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');history=[];start=time.time()
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=32,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42))
    for epoch in range(epochs):
        model.train();train_loss=0
        for bx,by in loader:
            optimizer.zero_grad();loss=loss_fn(model(bx),by);loss.backward();optimizer.step();train_loss+=float(loss.detach())*len(by)
        model.eval()
        with torch.no_grad():
            logits=model(vx);val_loss=float(loss_fn(logits,vy));val_acc=float((logits.argmax(1)==vy).float().mean())
        history.append({'epoch':epoch+1,'train_loss':train_loss/len(y),'validation_loss':val_loss,'validation_accuracy':val_acc})
        if val_loss<best:
            best=val_loss;best_epoch=epoch+1;torch.save(model.state_dict(),out/'weights.pt')
        print(f'Epoch {epoch+1}/{epochs}: train={train_loss/len(y):.4f} val={val_loss:.4f} val_acc={val_acc:.3f}',flush=True)
    model.load_state_dict(torch.load(out/'weights.pt',weights_only=True,map_location='cpu'));model.eval()
    with torch.no_grad():scores=torch.softmax(model(tx),dim=1).numpy()
    torch.onnx.export(model,torch.zeros(1,1,64,64),str(out/'model.onnx'),input_names=['image'],output_names=['logits'],dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}},opset_version=17,dynamo=False)
    import onnxruntime as ort
    session=ort.InferenceSession(str(out/'model.onnx'),providers=['CPUExecutionProvider'])
    exported=session.run(None,{'image':tx[:8].numpy()})[0]
    with torch.no_grad():expected=model(tx[:8]).numpy()
    error=float(np.max(np.abs(exported-expected)));assert error<1e-4
    baseline=json.loads((ROOT/'models/ct/metadata.json').read_text())
    meta={**baseline,'model_version':'iqoth-cnn-v1','architecture':'3 convolution blocks + dense classifier','framework':'PyTorch training / ONNX Runtime inference','parameters':sum(p.numel() for p in model.parameters()),'training_epochs':epochs,'best_epoch':best_epoch,'batch_size':32,'learning_rate':.001,'optimizer':'Adam','loss':'class-weighted cross entropy','image_size':64,'deep_learning':True,'stage_supported':False,'weights_sha256':hashlib.sha256((out/'model.onnx').read_bytes()).hexdigest(),'preprocessing':'EXIF orientation; grayscale; crop 8% from each edge; bilinear 64x64; divide by 255','torch_version':torch.__version__,'onnxruntime_version':ort.__version__}
    for k in ['gamma','regularization']:meta.pop(k,None)
    result={'images':len(manifest),'split_sizes':{s:len(v[1]) for s,v in data.items()},'class_counts':{c:sum(r['label']==c for r in manifest) for c in CLASSES},'selection':'Lowest class-weighted validation cross entropy across 40 epochs; fixed manifest. Test evaluated after checkpoint selection.','limitation':baseline['split_limitation'],'test':evaluate(ty.numpy(),scores,CLASSES),'history':history,'onnx_max_absolute_logit_error':error,'training_seconds':time.time()-start}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));(out/'evaluation.json').write_text(json.dumps(result,indent=2))
    for name in ['metadata.json','evaluation.json']:shutil.copyfile(out/name,ROOT/'web/real_samples'/name)
    print(json.dumps({'metadata':meta,'test':result['test'],'training_seconds':result['training_seconds']},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--epochs',type=int,default=40)
    a=p.parse_args();train(a.source.resolve(),a.epochs)
