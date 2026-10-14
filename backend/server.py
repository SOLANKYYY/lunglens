"""Local development server. Serves the UI and an image API on one origin.
Do not expose this standard-library development server publicly.
"""
import argparse
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from backend.inference import Predictor

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 10 * 1024 * 1024

class Handler(SimpleHTTPRequestHandler):
    predictor = None
    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()
    def send_json(self, code, data):
        payload = json.dumps(data).encode()
        self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
    def do_GET(self):
        route = urlsplit(self.path).path
        if route == '/api/health':
            self.send_json(200, {'ready': self.predictor is not None, 'metadata': self.predictor.meta if self.predictor else None}); return
        if route.startswith('/api/'):
            self.send_json(404, {'error':'Unknown API route'}); return
        super().do_GET()
    def do_POST(self):
        if urlsplit(self.path).path != '/api/predict':
            self.send_json(404, {'error':'Unknown API route'}); return
        origin = self.headers.get('Origin')
        if origin and origin not in (f'http://localhost:{self.server.server_port}', f'http://127.0.0.1:{self.server.server_port}'):
            self.send_json(403, {'error':'Local same-origin requests only.'}); return
        if not self.predictor:
            self.send_json(503, {'error':'No model loaded. Train a model and restart the API.'}); return
        try: length = int(self.headers.get('Content-Length', '0'))
        except ValueError: length = 0
        if not 0 < length <= LIMIT:
            self.send_json(413, {'error':'Provide an image under 10 MB.'}); return
        if self.headers.get('Content-Type','').split(';')[0] not in ('image/png','image/jpeg','image/webp'):
            self.send_json(415, {'error':'Use PNG, JPEG or WEBP.'}); return
        try:
            result = self.predictor.predict(self.rfile.read(length))
            self.send_json(200, result)
        except Exception:
            self.send_json(422, {'error':'Image could not be processed. Check file type, size and model compatibility.'})
    def list_directory(self, path):
        self.send_error(403, 'Directory browsing disabled')
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', default=str(ROOT/'models/demo'))
    args = parser.parse_args()
    try: Handler.predictor = Predictor(args.model)
    except FileNotFoundError: print('Model missing: UI will run without inference. See README.')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT/'web')))
    print(f'LungLens: http://127.0.0.1:{args.port} | Ctrl+C to stop', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
