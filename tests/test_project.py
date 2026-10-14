import csv
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
from PIL import Image
from backend.common import load_manifest, preprocess
from backend.inference import Predictor
from backend.server import Handler
ROOT=Path(__file__).resolve().parents[1]

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor=Predictor(ROOT/'models/demo')
        cls.sample=(ROOT/'web/demo_samples/pattern_round.png').read_bytes()
    def test_manifest_and_disjoint_split_sizes(self):
        data,classes,kind=load_manifest(ROOT/'datasets/synthetic/manifest.csv')
        self.assertEqual({k:len(v[1]) for k,v in data.items()},{'train':108,'val':36,'test':36})
        self.assertEqual(kind,'synthetic');self.assertEqual(len(classes),3)
    def test_inference_returns_finite_scores_and_no_stage(self):
        result=self.predictor.predict(self.sample)
        self.assertAlmostEqual(sum(result['scores'].values()),1,places=5)
        self.assertTrue(all(np.isfinite(list(result['scores'].values()))))
        self.assertEqual(result['data_kind'],'synthetic');self.assertIsNone(result['stage'])
        self.assertFalse(result['clinical_validation'])
    def test_invalid_image_is_rejected(self):
        with self.assertRaises(Exception):preprocess(b'not an image')
    def test_patient_overlap_is_rejected(self):
        # Construct two different images belonging to the same subject in different splits.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            Image.new('L',(32,32),20).save(root/'a.png');Image.new('L',(32,32),80).save(root/'b.png')
            rows=[{'path':'a.png','label':'pattern_none','patient_id':'same','split':'train','source':'synthetic'}, {'path':'b.png','label':'pattern_none','patient_id':'same','split':'test','source':'synthetic'}]
            with (root/'manifest.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Patient leakage'):load_manifest(root/'manifest.csv')
    def test_duplicate_pixels_across_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);Image.new('L',(32,32),20).save(root/'a.png');Image.new('L',(32,32),20).save(root/'b.png')
            rows=[{'path':'a.png','label':'pattern_none','patient_id':'a','split':'train','source':'synthetic'}, {'path':'b.png','label':'pattern_none','patient_id':'b','split':'test','source':'synthetic'}]
            with (root/'manifest.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Duplicate preprocessed'):load_manifest(root/'manifest.csv')
    def test_http_upload_and_validation(self):
        class QuietHandler(Handler):
            def log_message(self,*args):pass
        QuietHandler.predictor=self.predictor
        server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(ROOT/'web')))
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base+'/') as r:self.assertIn(b'LungLens',r.read())
            with urllib.request.urlopen(base+'/api/health') as r:self.assertTrue(json.load(r)['ready'])
            req=urllib.request.Request(base+'/api/predict',data=self.sample,headers={'Content-Type':'image/png','Origin':base})
            with urllib.request.urlopen(req) as r:
                result=json.load(r);self.assertIsNone(result['stage']);self.assertEqual(result['framework'],'numpy')
            for payload,headers,code in [(b'bad',{'Content-Type':'image/png'},422),(b'bad',{'Content-Type':'text/plain'},415),(self.sample,{'Content-Type':'image/png','Origin':'https://untrusted.invalid'},403)]:
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(urllib.request.Request(base+'/api/predict',data=payload,headers=headers))
                self.assertEqual(caught.exception.code,code)
            QuietHandler.predictor=None
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(urllib.request.Request(base+'/api/predict',data=self.sample,headers={'Content-Type':'image/png'}))
            self.assertEqual(caught.exception.code,503)
        finally:server.shutdown();server.server_close();worker.join()

if __name__=='__main__':unittest.main()
