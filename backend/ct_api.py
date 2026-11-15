"""Shared public inference handler. Upload bytes are processed in memory, never saved."""
import json
from http.server import BaseHTTPRequestHandler
from functools import lru_cache
from backend.ct_model import CTPredictor

@lru_cache(maxsize=1)
def predictor():
    return CTPredictor()

class CTHandler(BaseHTTPRequestHandler):
    def send_json(self, code, data):
        body=json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.split('?')[0]!='/api/health':
            self.send_json(404,{'error':'Unknown endpoint'}); return
        try: self.send_json(200,{'ready':True,'metadata':predictor().meta})
        except Exception: self.send_json(503,{'error':'Model unavailable','ready':False})

    def do_POST(self):
        if self.path.split('?')[0]!='/api/predict':
            self.send_json(404,{'error':'Unknown endpoint'}); return
        try: length=int(self.headers.get('Content-Length','0'))
        except ValueError: length=0
        if not 0<length<=4*1024*1024:
            self.send_json(413,{'error':'Choose an image between 1 byte and 4 MB.'}); return
        if self.headers.get('Content-Type','').split(';')[0] not in ('image/png','image/jpeg','image/webp'):
            self.send_json(415,{'error':'Use PNG, JPEG or WEBP.'}); return
        try: result=predictor().predict(self.rfile.read(length))
        except (ValueError, OSError) as error:
            self.send_json(422,{'error':str(error)}); return
        except Exception:
            self.send_json(503,{'error':'Model unavailable. Please retry later.'}); return
        self.send_json(200,result)
