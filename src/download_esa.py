"""Parallel, resumable download of one ESA Anomaly Dataset mission from Zenodo.

Zenodo throttles each connection (~30 KB/s), so the file is fetched as byte
ranges over many connections. Finished chunks are kept, so re-running resumes.
The assembled zip is checked against Zenodo's MD5.

usage: download_esa.py <Mission1|Mission2|Mission3> <target_dir> [workers]
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

RECORD = "https://zenodo.org/api/records/15237121"
CHUNK = 16 * 1024 * 1024


def fetch_range(url, start, end, path, retries=200):
    """Fetch bytes start..end into `path`; an interrupted .part file is continued, not restarted."""
    expected = end - start + 1
    part = path.with_suffix(".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have > expected:  # corrupt leftover
            part.unlink()
            have = 0
        if have < expected:
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                    while True:
                        buf = r.read(1 << 16)
                        if not buf:
                            break
                        f.write(buf)
            except Exception:
                time.sleep(min(60, 5 * (attempt % 6 + 1)))
                continue
        if part.stat().st_size == expected:
            part.replace(path)
            return expected
    raise RuntimeError(f"chunk {path.name} failed")


def main(mission, target, workers=32):
    target = Path(target)
    chunks_dir = target / f"ESA-{mission}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen(RECORD, timeout=60))
    info = next(f for f in meta["files"] if f["key"] == f"ESA-{mission}.zip")
    url, size, md5 = info["links"]["self"], info["size"], info["checksum"].split(":")[1]
    ranges = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK))]
    todo = [r for r in ranges if not (chunks_dir / f"{r[0]:05d}").exists()]
    print(f"{mission}: {size / 1e9:.2f} GB, {len(ranges)} chunks, {len(todo)} to fetch, {workers} workers", flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_range, url, s, e, chunks_dir / f"{i:05d}") for i, s, e in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            done += fut.result()
            if k % 10 == 0 or k == len(futs):
                rate = done / (time.time() - t0)
                left = (sum(e - s + 1 for _, s, e in todo) - done) / max(rate, 1)
                print(f"  {k}/{len(todo)} chunks, {rate / 1024:.0f} KB/s, ~{left / 60:.0f} min left", flush=True)

    out = target / f"ESA-{mission}.zip"
    h = hashlib.md5()
    with open(out, "wb") as f:
        for i, _, _ in ranges:
            data = (chunks_dir / f"{i:05d}").read_bytes()
            h.update(data)
            f.write(data)
    ok = h.hexdigest() == md5
    print(f"assembled {out} ({out.stat().st_size:,} bytes), MD5 {'OK' if ok else 'MISMATCH'}", flush=True)
    if ok:
        for p in chunks_dir.iterdir():
            p.unlink()
        chunks_dir.rmdir()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32)
