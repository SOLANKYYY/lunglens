"""Train a transparent baseline with NumPy, PyTorch, or TensorFlow.
Run from project root: python -m backend.train --framework numpy
This is a 32x32 image classifier, not a nodule detector or staging model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from backend.common import load_manifest, softmax, evaluate, IMAGE_SIZE


def train(args):
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    grouped, classes, kind = load_manifest(args.manifest)
    x, y = grouped['train']; vx, vy = grouped['val']; tx, ty = grouped['test']
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    history = []; best_loss = float('inf'); framework = args.framework
    # Same single dense softmax baseline across frameworks for an explainable comparison.
    # Only train sees gradient updates; val selects checkpoint; test is evaluated once below.
    if framework == 'numpy':
        rng = np.random.default_rng(42)
        weight = rng.normal(0, .01, (x.shape[1], len(classes))).astype('float32'); bias = np.zeros(len(classes), dtype='float32')
        targets = np.eye(len(classes), dtype='float32')[y]
        for epoch in range(args.epochs):
            scores = softmax(x @ weight + bias)
            grad = (scores-targets)/len(x)
            weight -= .08*(x.T @ grad + .001*weight); bias -= .08*grad.sum(axis=0)
            vs = softmax(vx @ weight + bias)
            loss = float(-np.log(vs[np.arange(len(vy)), vy]+1e-8).mean())
            history.append({'epoch': epoch+1, 'val_loss': loss})
            if loss < best_loss:
                best_loss = loss; np.savez(out/'weights.npz', weight=weight, bias=bias)
        weights = np.load(out/'weights.npz', allow_pickle=False)
        test_scores = softmax(tx @ weights['weight'] + weights['bias'])
    elif framework == 'pytorch':
        import torch
        from torch import nn
        torch.manual_seed(42)
        model = nn.Linear(IMAGE_SIZE*IMAGE_SIZE, len(classes))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)
        xx = torch.from_numpy(x); yy = torch.from_numpy(y)
        for epoch in range(args.epochs):
            model.train(); optimizer.zero_grad(); loss = nn.functional.cross_entropy(model(xx), yy); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(nn.functional.cross_entropy(model(torch.from_numpy(vx)), torch.from_numpy(vy)))
            history.append({'epoch': epoch+1, 'val_loss': val_loss})
            if val_loss < best_loss:
                best_loss = val_loss; torch.save(model.state_dict(), out/'weights.pt')
        model.load_state_dict(torch.load(out/'weights.pt', weights_only=True, map_location='cpu')); model.eval()
        with torch.no_grad(): test_scores = torch.softmax(model(torch.from_numpy(tx)), dim=1).numpy()
    else:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential([tf.keras.Input(shape=(IMAGE_SIZE*IMAGE_SIZE,)), tf.keras.layers.Dense(len(classes))])
        model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        checkpoint = tf.keras.callbacks.ModelCheckpoint(str(out/'weights.keras'), monitor='val_loss', save_best_only=True)
        run = model.fit(x, y, validation_data=(vx, vy), epochs=args.epochs, batch_size=32, callbacks=[checkpoint], verbose=2)
        history = [{'epoch': i+1, 'val_loss': float(loss)} for i, loss in enumerate(run.history['val_loss'])]
        model = tf.keras.models.load_model(out/'weights.keras', compile=False)
        test_scores = softmax(model.predict(tx, verbose=0))
    metadata = {'framework': framework, 'classes': classes, 'data_kind': kind, 'image_size': IMAGE_SIZE, 'architecture': 'linear-softmax-baseline', 'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), 'clinical_validation': False, 'stage_supported': False, 'seed': 42}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    result = {'data_kind': kind, 'warning': 'Synthetic scores measure artificial pattern recognition, not cancer detection.' if kind=='synthetic' else 'Research evaluation only; no clinical validation or cancer staging.', 'split_sizes': {k:len(v[1]) for k,v in grouped.items()}, 'selection': 'Minimum validation loss; test evaluated after checkpoint selection', 'test': evaluate(ty, test_scores, classes), 'history': history}
    (out/'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='history'}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['numpy','pytorch','tensorflow'], default='numpy')
    parser.add_argument('--manifest', default='datasets/synthetic/manifest.csv')
    parser.add_argument('--output', default='models/demo')
    parser.add_argument('--epochs', type=int, default=200)
    train(parser.parse_args())
