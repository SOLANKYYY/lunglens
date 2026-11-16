import hashlib, io, json, threading, unittest, urllib.request, urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
from PIL import Image
from backend.ct_api import CTHandler
from backend.ct_model import CTPredictor
from backend.cnn_inference import CNNPredictor
ROOT=Path(__file__).resolve().parents[1]
class RealCTTests(unittest.TestCase):
    def test_restored_model_and_samples(self):
        model=CTPredictor(); metadata=model.meta
        self.assertEqual(hashlib.sha256((ROOT/'models/ct/weights.npz').read_bytes()).hexdigest(),metadata['weights_sha256'])
        for sample in json.loads((ROOT/'web/real_samples/samples.json').read_text()):
            data=(ROOT/'web/real_samples'/sample['file']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),sample['sha256'])
            output=model.predict(data)
            self.assertTrue(np.isfinite(list(output['scores'].values())).all())
            self.assertIsNone(output['stage']); self.assertFalse(output['clinical_validation'])
    def test_no_similarity_group_or_exact_file_overlap(self):
        rows=json.loads((ROOT/'models/ct/manifest.json').read_text());groups={};hashes={}
        self.assertEqual(len(rows),1097)
        for row in rows:
            for mapping,key in [(groups,row['similarity_group']),(hashes,row['sha256'])]:
                self.assertEqual(mapping.setdefault(key,row['split']),row['split'])
    def test_reject_invalid_and_blank_images(self):
        model=CTPredictor()
        with self.assertRaises(Exception):model.predict(b'not an image')
        out=io.BytesIO();Image.new('L',(128,128)).save(out,format='PNG')
        with self.assertRaises(ValueError):model.predict(out.getvalue())
    def test_http_inference_and_errors(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),CTHandler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base+'/api/health') as r:self.assertTrue(json.load(r)['ready'])
            data=(ROOT/'web/real_samples/normal-1.jpg').read_bytes()
            req=urllib.request.Request(base+'/api/predict',data=data,headers={'Content-Type':'image/jpeg'})
            with urllib.request.urlopen(req) as r:actual=json.load(r)
            self.assertEqual(actual,CNNPredictor().predict(data))
            for mime,body,status in [('text/plain',b'x',415),('image/png',b'bad',422),('image/png',b'',413)]:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(urllib.request.Request(base+'/api/predict',data=body,headers={'Content-Type':mime}))
                self.assertEqual(error.exception.code,status)
        finally:server.shutdown();server.server_close();thread.join()
