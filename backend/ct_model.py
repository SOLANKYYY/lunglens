"""Real CT research baseline. All runtime preprocessing lives here."""
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

CLASSES = ['normal', 'benign', 'malignant']
SIZE = 64

def features(data):
    with Image.open(io.BytesIO(data)) as im:
        if im.format not in ('PNG', 'JPEG', 'WEBP'):
            raise ValueError('Use a PNG, JPEG or WEBP CT slice, not DICOM.')
        if im.width * im.height > 20_000_000 or min(im.size) < 64:
            raise ValueError('Image must be at least 64 pixels per side and under 20 megapixels.')
        im = ImageOps.exif_transpose(im).convert('L')
        # Central crop reduces borders and burnt-in peripheral text; identical in training/inference.
        w, h = im.size
        im = im.crop((int(w*.08), int(h*.08), int(w*.92), int(h*.92)))
        x = np.asarray(im.resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float64)/255
        if x.std() < .02:
            raise ValueError('Image has too little contrast to process.')
        return x.reshape(-1)

def kernel(a, b, gamma):
    distance = np.maximum((a*a).sum(1)[:,None] + (b*b).sum(1)[None,:] - 2*a@b.T, 0)/a.shape[1]
    return np.exp(-gamma*distance)

class CTPredictor:
    def __init__(self, directory=None):
        directory = Path(directory or Path(__file__).resolve().parents[1]/'models/ct')
        self.meta = json.loads((directory/'metadata.json').read_text())
        with np.load(directory/'weights.npz', allow_pickle=False) as w:
            self.reference, self.alpha = w['reference'], w['alpha']
        self.gamma = self.meta['gamma']

    def predict(self, data):
        x = features(data)[None,:]
        raw = (kernel(x, self.reference, self.gamma) @ self.alpha)[0]
        # Raw kernel-regression outputs, deliberately not probabilities or cancer risk.
        return {'class': CLASSES[int(raw.argmax())], 'scores': dict(zip(CLASSES, map(float,raw))),
                'score_type':'kernel regression class score (not a probability)',
                'framework':'NumPy', 'data_kind':'research', 'model_version':self.meta['model_version'],
                'stage':None, 'stage_status':'Not assessed: dataset has no per-image stage labels.',
                'clinical_validation':False,
                'warning':'Experimental CT image classification only. Scores do not measure cancer probability, severity or stage. A normal prediction does not rule out cancer. Non-CT inputs can receive a class.'}
