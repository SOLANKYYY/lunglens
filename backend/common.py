"""Shared, identical image preprocessing and split validation for every model."""
import csv
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 32
SYNTHETIC_CLASSES = ['pattern_none', 'pattern_round', 'pattern_irregular']
RESEARCH_CLASSES = ['normal', 'benign', 'malignant']
Image.MAX_IMAGE_PIXELS = 20_000_000

def preprocess(data):
    """8-bit PNG/JPEG/WEBP -> grayscale, 32 x 32, flattened [0,1]. No DICOM."""
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in ('PNG', 'JPEG', 'WEBP'):
            raise ValueError('Use PNG, JPEG or WEBP; DICOM requires a separate CT preprocessing pipeline.')
        if image.width * image.height > 20_000_000:
            raise ValueError('Image exceeds 20 megapixels.')
        image = ImageOps.exif_transpose(image).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0

def load_manifest(path):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('Manifest is empty.')
    required = {'path', 'label', 'patient_id', 'split', 'source'}
    if not required.issubset(rows[0]):
        raise ValueError('Required columns: ' + ', '.join(sorted(required)))
    sources = {r['source'] for r in rows}
    if sources == {'synthetic'}:
        classes, kind = SYNTHETIC_CLASSES, 'synthetic'
    elif 'synthetic' in sources:
        raise ValueError('Do not mix synthetic and real data.')
    else:
        classes, kind = RESEARCH_CLASSES, 'research'
    patients, pixels, seen_paths = {}, {}, set()
    grouped = {split: ([], []) for split in ('train', 'val', 'test')}
    for row in rows:
        if row['label'] not in classes or row['split'] not in grouped:
            raise ValueError('Unknown label or split: ' + str(row))
        if not row['patient_id'].strip() or not row['source'].strip():
            raise ValueError('Every image needs a verified patient/group ID and source.')
        patient = row['patient_id']
        if patient in patients and patients[patient] != row['split']:
            raise ValueError('Patient leakage detected: ' + patient)
        patients[patient] = row['split']
        image_path = (path.parent / row['path']).resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError('Image path must remain under the manifest directory.')
        if image_path in seen_paths:
            raise ValueError('Duplicate path in manifest: ' + row['path'])
        seen_paths.add(image_path)
        x = preprocess(image_path.read_bytes())
        digest = hashlib.sha256(x.tobytes()).hexdigest()
        if digest in pixels and pixels[digest] != row['split']:
            raise ValueError('Duplicate preprocessed image across splits: ' + row['path'])
        pixels[digest] = row['split']
        grouped[row['split']][0].append(x)
        grouped[row['split']][1].append(classes.index(row['label']))
    for split, (xs, ys) in grouped.items():
        if set(ys) != set(range(len(classes))):
            raise ValueError(f'{split} must contain every class.')
        grouped[split] = (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64))
    return grouped, classes, kind

def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

def evaluate(y, scores, classes):
    pred = scores.argmax(axis=1)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, guessed in zip(y, pred):
        matrix[actual, guessed] += 1
    per_class = {}
    for i, label in enumerate(classes):
        tp = int(matrix[i, i]); precision = tp / max(1, int(matrix[:, i].sum())); recall = tp / max(1, int(matrix[i].sum()))
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': 2*precision*recall/max(1e-12, precision+recall), 'support': int(matrix[i].sum())}
    return {'accuracy': float(np.mean(y == pred)), 'macro_f1': float(np.mean([v['f1'] for v in per_class.values()])), 'confusion_matrix_rows_actual_columns_predicted': matrix.tolist(), 'class_order': classes, 'per_class': per_class}
