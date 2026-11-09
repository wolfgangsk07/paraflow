#!/usr/bin/env python3
"""Measure ParaFlow/llama.cpp decode throughput at fixed concurrency.

The server must already be running. For each concurrency level this script:
  1. sends a warm-up batch and lets it finish,
  2. waits for a cooldown,
  3. sends the formal batch,
  4. starts a shared window after every slot has produced its first token,
  5. counts all streamed tokens inside that window.

It is intentionally model-agnostic: point it at a running llama-server and pass
the model's prompt/output settings on the command line.
"""
from __future__ import annotations

import argparse
import csv
import http.client
import json
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any


DEFAULT_PROMPT = (
    "Write a concise technical note about why GPU utilization matters for "
    "local language model inference. Mention batching, KV cache and pipeline "
    "stages."
)


@dataclass
class Point:
    concurrency: int
    status: str
    tps: float | None
    window_s: float | None
    admission_s: float | None
    first_spread_s: float | None
    finished_before_end: int
    warm_tokens: int
    formal_tokens: int
    requests: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-concurrency llama.cpp decode benchmark")
    p.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server base URL")
    p.add_argument("--concurrency", default="1,16,32,48,64,80", help="comma-separated C values")
    p.add_argument("--n-predict", type=int, default=256, help="formal output tokens per request")
    p.add_argument("--warm-tokens", type=int, default=432, help="warm-up output tokens per request; 0 disables")
    p.add_argument("--window", type=float, default=5.0, help="measurement window seconds")
    p.add_argument("--cooldown", type=float, default=60.0, help="seconds between warm-up and formal")
    p.add_argument("--timeout", type=float, default=900.0, help="per-request read timeout")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=74000)
    p.add_argument("--api-key", default=None, help="optional bearer token; defaults to LLAMA_API_KEY")
    p.add_argument("--no-mtp", action="store_true", help="do not request MTP/speculative decoding")
    p.add_argument("--json-out", default=None)
    p.add_argument("--csv-out", default=None)
    return p.parse_args()


def endpoint(base_url: str) -> tuple[str, int, str]:
    from urllib.parse import urlparse
    u = urlparse(base_url if "://" in base_url else "http://" + base_url)
    if u.scheme not in ("http", "https"):
        raise SystemExit("base-url must be http:// or https://")
    return u.scheme, u.port or (443 if u.scheme == "https" else 80), u.hostname or "127.0.0.1"


def request_body(args: argparse.Namespace, slot: int, n_predict: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt": args.prompt,
        "id_slot": slot,
        "temperature": 0.0,
        "top_k": 40,
        "seed": args.seed + slot,
        "n_predict": n_predict,
        "ignore_eos": False,
        "cache_prompt": True,
        "stream": True,
        "return_tokens": True,
    }
    if not args.no_mtp:
        body["mtp"] = True
        body["speculative.n_max"] = 3
    return body


def stream_once(
    host: str,
    port: int,
    tls: bool,
    args: argparse.Namespace,
    slot: int,
    n_predict: int,
    shared: dict[str, Any],
    barrier: threading.Barrier,
    cancel_after_window: bool,
) -> dict[str, Any]:
    body = json.dumps(request_body(args, slot, n_predict)).encode()
    headers = {"Content-Type": "application/json"}
    key = args.api_key or None
    if key:
        headers["Authorization"] = "Bearer " + key
    path = "/completion"
    conn_cls = http.client.HTTPSConnection if tls else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=args.timeout)
    events: list[tuple[float, int]] = []
    first: float | None = None
    predicted = 0
    finished = False
    try:
        barrier.wait(timeout=120)
        conn.request("POST", path, body, headers)
        resp = conn.getresponse()
        if resp.status != 200:
            detail = resp.read(500).decode(errors="replace")
            raise RuntimeError(f"HTTP {resp.status}: {detail}")
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            try:
                obj = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue
            if "error" in obj:
                raise RuntimeError(str(obj["error"]))
            now = time.monotonic()
            n = int(obj.get("tokens_predicted", predicted))
            if n > predicted:
                if first is None:
                    first = now
                    with shared["lock"]:
                        shared["first"].append(now)
                        if len(shared["first"]) == shared["C"]:
                            shared["end"] = max(shared["first"]) + 0.5 + args.window
                events.append((now, n - predicted))
                predicted = n
            if obj.get("stop"):
                finished = True
                break
            if cancel_after_window and shared.get("end") and now > shared["end"]:
                break
        return {"slot": slot, "first": first, "events": events, "predicted": predicted, "finished": finished}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_batch(host: str, port: int, tls: bool, args: argparse.Namespace, c: int, n_predict: int, cancel: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shared: dict[str, Any] = {"lock": threading.Lock(), "first": [], "C": c}
    barrier = threading.Barrier(c)
    with ThreadPoolExecutor(max_workers=c) as ex:
        rows = list(ex.map(lambda s: stream_once(host, port, tls, args, s, n_predict, shared, barrier, cancel), range(c)))
    return rows, shared


def measure_point(args: argparse.Namespace, c: int) -> Point:
    tls = args._tls
    host, port = args._host, args._port
    if args.warm_tokens > 0:
        print(f"[C{c}] warm-up {args.warm_tokens} tokens x {c}", file=sys.stderr, flush=True)
        run_batch(host, port, tls, args, c, args.warm_tokens, cancel=False)
        if args.cooldown > 0:
            print(f"[C{c}] cooldown {args.cooldown:g}s", file=sys.stderr, flush=True)
            time.sleep(args.cooldown)
    print(f"[C{c}] formal {args.n_predict} tokens x {c}", file=sys.stderr, flush=True)
    t0 = time.monotonic()
    rows, shared = run_batch(host, port, tls, args, c, args.n_predict, cancel=True)
    firsts = [r["first"] for r in rows if r["first"] is not None]
    if len(firsts) != c:
        return Point(c, "missing_first_token", None, None, None, None, 0, args.warm_tokens, args.n_predict, c)
    start = max(firsts) + 0.5
    end = start + args.window
    events = [(t, n) for r in rows for t, n in r["events"] if start <= t < end]
    counts = sum(n for _, n in events)
    return Point(
        concurrency=c,
        status="measured",
        tps=round(counts / args.window, 3),
        window_s=args.window,
        admission_s=round(max(firsts) - t0, 2),
        first_spread_s=round(max(firsts) - min(firsts), 2),
        finished_before_end=sum(1 for r in rows if r["finished"] and (r["events"][-1][0] < end if r["events"] else False)),
        warm_tokens=args.warm_tokens,
        formal_tokens=args.n_predict,
        requests=c,
    )


def main() -> int:
    args = parse_args()
    args._tls, args._port, args._host = endpoint(args.base_url)
    args._tls = args._tls == "https"
    args.api_key = args.api_key or __import__("os").environ.get("LLAMA_API_KEY")
    try:
        cs = [int(x) for x in args.concurrency.split(",") if x.strip()]
    except ValueError:
        raise SystemExit("--concurrency must be comma-separated integers")
    if not cs or min(cs) < 1:
        raise SystemExit("--concurrency must contain positive integers")
    points = []
    for c in cs:
        try:
            p = measure_point(args, c)
        except Exception as e:
            print(f"[C{c}] failed: {e}", file=sys.stderr, flush=True)
            p = Point(c, f"failed: {e}"[:200], None, None, None, None, 0, args.warm_tokens, args.n_predict, c)
        points.append(p)
        print(json.dumps(asdict(p), ensure_ascii=False), flush=True)
    result = {"base_url": args.base_url, "points": [asdict(p) for p in points], "protocol": {
        "warm_tokens": args.warm_tokens,
        "formal_tokens": args.n_predict,
        "window_s": args.window,
        "cooldown_s": args.cooldown,
        "mtp": not args.no_mtp,
    }}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    if args.csv_out:
        with open(args.csv_out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(points[0]).keys()))
            w.writeheader()
            for p in points:
                w.writerow(asdict(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
