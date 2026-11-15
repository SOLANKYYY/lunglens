"""Serve the real CT app locally: python -m backend.ct_server."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from backend.ct_api import CTHandler
class Handler(CTHandler, SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/'): CTHandler.do_GET(self)
        else: SimpleHTTPRequestHandler.do_GET(self)
    def list_directory(self,path):
        self.send_error(403); return None
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'web'
    print('LungLens: http://127.0.0.1:8000',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8000),partial(Handler,directory=str(root))).serve_forever()
