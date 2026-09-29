"""Page locale de vérification humaine de l'audit xLAM.

    python3 audit_xlam/verifier.py        # puis http://localhost:8765

Lit et réécrit audit_xlam/a_verifier.csv (colonnes verdict_humain, commentaire).
Bibliothèque standard uniquement. Vérification à l'aveugle : l'avis du juge n'est
affiché qu'après avoir donné le sien, sinon l'accord humain/juge serait biaisé.
"""

import csv
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ICI = Path(__file__).resolve().parent
CSV = ICI / "a_verifier.csv"
PAGE = ICI / "verifier.html"
PORT = int(os.environ.get("PORT", 8765))
VERDICTS = {"non_derivable", "derivable", "incertain", ""}


def lire():
    with CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def ecrire(colonnes, lignes):
    # Écriture atomique : un crash en cours d'écriture ne corrompt pas le CSV.
    fd, tmp = tempfile.mkstemp(dir=ICI, suffix=".csv")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonnes, lineterminator="\n")  # comme pandas
        w.writeheader()
        w.writerows(lignes)
    os.replace(tmp, CSV)


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/cas":
            _, lignes = lire()
            self._json(200, [l for l in lignes if l["a_verifier"]])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/verdict":
            return self.send_error(404)
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if data.get("verdict_humain", "") not in VERDICTS:
            return self._json(400, {"erreur": "verdict inconnu"})
        colonnes, lignes = lire()
        for l in lignes:
            if l["xlam_id"] == str(data["xlam_id"]):
                l["verdict_humain"] = data.get("verdict_humain", "")
                l["commentaire"] = data.get("commentaire", "")
                break
        else:
            return self._json(404, {"erreur": "xlam_id introuvable"})
        ecrire(colonnes, lignes)
        self._json(200, {"ok": True})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Vérification xLAM : http://localhost:{PORT}  (CSV : {CSV})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
