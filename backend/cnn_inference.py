"""Restored CNN inference and plain-language class explanations."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from backend.ct_model import features,CLASSES
MEANINGS={
 'normal':'The CNN gives the highest score to the normal-labelled training category. This does not confirm a healthy lung or exclude cancer.',
 'benign':'The CNN gives the highest score to the benign-labelled category, which represents non-cancerous cases in this dataset. The prediction cannot prove that a lesion is benign.',
 'malignant':'The CNN gives the highest score to the malignant-labelled category, which represents cancerous cases in this dataset. This is a research prediction, not a confirmed cancer diagnosis.'}
class CNNPredictor:
    def __init__(self,directory=None):
        directory=Path(directory or Path(__file__).resolve().parents[1]/'models/cnn')
        self.meta=json.loads((directory/'metadata.json').read_text())
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(directory/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    def predict(self,data):
        x=features(data).astype('float32').reshape(1,1,64,64)
        logits=self.session.run(None,{'image':x})[0][0].astype('float64')
        exp=np.exp(logits-logits.max());scores=exp/exp.sum();order=np.argsort(scores)[::-1]
        label=CLASSES[int(order[0])];lead=float((scores[order[0]]-scores[order[1]])*100)
        return {'class':label,'scores':dict(zip(CLASSES,map(float,scores))),
            'score_type':'uncalibrated CNN softmax class scores','framework':'PyTorch / ONNX Runtime',
            'data_kind':'research','model_version':self.meta['model_version'],'clinical_validation':False,
            'explanation':{'heading':label.capitalize()+'-pattern prediction','meaning':MEANINGS[label],
                'malignant_score_percent':float(scores[2]*100),'winning_score_percent':float(scores[order[0]]*100),
                'score_lead_percentage_points':lead,
                'score_meaning':'Softmax converts the CNN outputs into three scores that sum to 100%. These are uncalibrated model preferences, not the percentage of lung affected or a personal probability of cancer.',
                'method':'The CNN learns image features through convolution layers, then selects the class with the largest output score. This explanation describes the decision rule, not a localized tumor finding.',
                'next_step':'For a real medical scan, ask a radiologist or treating clinician to interpret the complete study and clinical findings. Do not make medical decisions from this prototype.'},
            'stage':None,'stage_status':'Stage cannot be determined from this single-image classifier. Clinical staging requires tumor, lymph-node and spread information.',
            'warning':'Educational CNN with limited test performance. Any class, including a high-scoring prediction, may be wrong. Scores are not cancer severity or affected-lung percentage. Non-CT images can receive scores.'}
