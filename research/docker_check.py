"""Verify the built serving image: CPU-only torch, offline startup, analysis and rejections.

    docker build -t veridoc-api:0.5 .
    .venv/bin/python -m research.docker_check            # -> results/docker_verification.json

The container runs with `--network none` and the checkpoint mounted read-only, so a
passing check also shows that serving needs no network. Requests are sent from inside
the container (there is no network to reach it from the host).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

from research.config import resolve

PROBE = r'''
import json, os, sys, urllib.request, urllib.error, uuid
def post(data, name="p.jpg", ctype="image/jpeg"):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/analyze", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        r = urllib.request.urlopen(req, timeout=600); return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
out = {}
h = urllib.request.urlopen("http://127.0.0.1:8000/health")
out["health"] = {"http_status": h.status, **json.loads(h.read())}
s, d = post(open("/tmp/page.jpg", "rb").read())
out["analyze_page"] = {"status": s, "n_regions": d["n_regions"], "regions": d["regions"][:3],
                       "risk_score": d["risk_score"], "keys": sorted(d), "timing_ms": d["timing_ms"]}
s, d = post(b"not an image" * 100, "x.png", "image/png"); out["non_image"] = {"status": s, **d}
# oversize: announce an 11 MB body and read the answer; the server rejects from the
# Content-Length header and closes without reading the body (urllib would see a reset)
import socket
sock = socket.create_connection(("127.0.0.1", 8000))
sock.sendall(b"POST /v1/analyze HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; "
             b"boundary=zz\r\nContent-Length: %d\r\n\r\n" % (11 * 1024 * 1024))
raw = sock.recv(65536).decode(errors="replace")
out["oversize_11MB"] = {"status": int(raw.split()[1]), "body": raw.split("\r\n\r\n", 1)[-1]}
print(json.dumps(out))
'''


def sh(*cmd: str, check: bool = True, **kw) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and p.returncode:
        raise RuntimeError(f"{cmd[:4]} failed: {p.stderr.strip()[-800:]}")
    return p.stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="veridoc-api:0.5")
    ap.add_argument("--checkpoint", default="artifacts/run2/best.pt")
    ap.add_argument("--page", default="data/generated/val/invoice_0010_splice_0.jpg")
    a = ap.parse_args(argv)
    name = "veridoc-check"
    insp = json.loads(sh("docker", "image", "inspect", a.image))[0]
    res = {"image": a.image, "image_id": insp["Id"], "created": insp["Created"],
           "image_size_bytes": insp.get("Size"), "docker_version": sh("docker", "version", "--format", "{{.Server.Version}}")}
    res["torch_in_image"] = sh("docker", "run", "--rm", "--network", "none", "--entrypoint", "python", a.image, "-c",
                               "import torch;print(torch.__version__, torch.cuda.is_available())")
    res["sklearn_in_image"] = "absent" if "ModuleNotFoundError" in subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", a.image, "-c", "import sklearn"],
        capture_output=True, text=True).stderr else "present"
    sh("docker", "rm", "-f", name, check=False)
    sh("docker", "run", "-d", "--name", name, "--network", "none",
       "-v", f"{resolve(a.checkpoint)}:/models/best.pt:ro", "-v", f"{resolve(a.page)}:/tmp/page.jpg:ro", a.image)
    try:
        for _ in range(120):
            ok = subprocess.run(["docker", "exec", name, "python", "-c",
                                 "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"],
                                capture_output=True).returncode == 0
            if ok:
                break
            time.sleep(1)
        res["network_inside_container"] = sh("docker", "exec", name, "python", "-c",
                                             "import socket\ntry:\n socket.create_connection(('pypi.org',443),timeout=3);print('reachable')\n"
                                             "except OSError as e:\n print('unreachable:', type(e).__name__)")
        res["probe"] = json.loads(sh("docker", "exec", "-i", name, "python", "-", input=PROBE))
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        res["container_log_tail"] = (logs.stdout + logs.stderr).strip().splitlines()[-8:]
    finally:
        sh("docker", "rm", "-f", name, check=False)
    res["page"] = a.page
    path = resolve("results/docker_verification.json")
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
