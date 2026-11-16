import hashlib,json,unittest
from pathlib import Path
from backend.cnn_inference import CNNPredictor
ROOT=Path(__file__).resolve().parents[1]
class CNNTests(unittest.TestCase):
    def test_saved_network_and_explanations(self):
        model=CNNPredictor()
        self.assertTrue(model.meta['deep_learning'])
        self.assertEqual(hashlib.sha256((ROOT/'models/cnn/model.onnx').read_bytes()).hexdigest(),model.meta['weights_sha256'])
        for name in ['normal-1.jpg','benign-1.jpg','malignant-1.jpg']:
            result=model.predict((ROOT/'web/real_samples'/name).read_bytes())
            self.assertAlmostEqual(sum(result['scores'].values()),1,places=6)
            self.assertTrue(all(0<=x<=1 for x in result['scores'].values()))
            self.assertEqual(result['class'],max(result['scores'],key=result['scores'].get))
            self.assertAlmostEqual(result['explanation']['malignant_score_percent'],result['scores']['malignant']*100)
            self.assertIsNone(result['stage']);self.assertFalse(result['clinical_validation'])
            self.assertIn('not',result['explanation']['score_meaning'])
    def test_staging_records_are_separate_historical_data(self):
        data=json.loads((ROOT/'web/staging-reference.json').read_text())
        self.assertEqual(len(data['records']),421)
        self.assertEqual(sum(data['counts'].values()),421)
        self.assertNotIn('IV',data['counts'])
        self.assertIn('Not predictions',data['purpose'])
