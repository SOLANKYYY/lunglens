"""Load only trusted, locally trained weights. Never upload or unpickle arbitrary models."""
import json
from pathlib import Path
import numpy as np
from backend.common import preprocess, softmax, IMAGE_SIZE

class Predictor:
    def __init__(self, directory):
        directory = Path(directory)
        self.meta = json.loads((directory/'metadata.json').read_text())
        if self.meta['image_size'] != IMAGE_SIZE or self.meta['architecture'] != 'linear-softmax-baseline':
            raise ValueError('Incompatible preprocessing or architecture.')
        fw = self.meta['framework']
        if fw == 'numpy':
            weights = np.load(directory/'weights.npz', allow_pickle=False)
            self.weight, self.bias = weights['weight'], weights['bias']
        elif fw == 'pytorch':
            import torch
            self.model = torch.nn.Linear(IMAGE_SIZE*IMAGE_SIZE, len(self.meta['classes']))
            self.model.load_state_dict(torch.load(directory/'weights.pt', weights_only=True, map_location='cpu'))
            self.model.eval()
        elif fw == 'tensorflow':
            import tensorflow as tf
            self.model = tf.keras.models.load_model(directory/'weights.keras', compile=False)
        else: raise ValueError('Unsupported framework')

    def predict(self, data):
        x = preprocess(data)[None, :]
        fw = self.meta['framework']
        if fw == 'numpy': scores = softmax(x @ self.weight + self.bias)[0]
        elif fw == 'pytorch':
            import torch
            with torch.no_grad(): scores = torch.softmax(self.model(torch.from_numpy(x)), dim=1).numpy()[0]
        else: scores = softmax(self.model.predict(x, verbose=0))[0]
        return {'class': self.meta['classes'][int(scores.argmax())], 'scores': {label: float(value) for label,value in zip(self.meta['classes'],scores)}, 'framework': fw, 'data_kind': self.meta['data_kind'], 'stage': None, 'stage_status': 'Not assessed: this model has no staging labels or clinical context.', 'clinical_validation': False, 'warning': 'Artificial-pattern classifier only. Not trained to detect cancer.' if self.meta['data_kind']=='synthetic' else 'Unvalidated research classifier. Scores are not calibrated cancer risk; arbitrary images may receive a class.'}
