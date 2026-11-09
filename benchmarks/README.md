# Benchmark client

`bench.py` measures decode throughput of a running `llama-server` at fixed
concurrency. It does not start the server or download a model.

## Protocol

For each concurrency `C`:

1. send `C` warm-up requests (`--warm-tokens`, default 432) and let them finish;
2. wait `--cooldown` seconds (default 60);
3. send `C` formal requests (`--n-predict`, default 256);
4. start the measurement window 0.5 s after every request has produced its
   first token;
5. count all streamed tokens in the next `--window` seconds (default 5).

The same prompt and `id_slot` values are reused so the warm-up also fills the
prompt cache.

## Example

Start a server with at least the largest `-np` you will test, then:

```sh
python3 benchmarks/bench.py \
  --base-url http://127.0.0.1:8080 \
  --concurrency 1,16,32,48,64,80 \
  --n-predict 256 \
  --warm-tokens 432 \
  --window 5 \
  --cooldown 60 \
  --json-out results.json \
  --csv-out results.csv
```

For a non-MTP model add `--no-mtp`. If the server uses an API key, pass
`--api-key` or set `LLAMA_API_KEY`.

## Notes

- `--concurrency` must not exceed the server's `-np`.
- This is a client only; benchmark numbers depend on the model, quantization,
  KV type, context, GPU clocks, cooling and server flags.
- The default 432-token warm-up is deliberately expensive: it matches the
  published measurement protocol, not a quick smoke test.
- On a busy or thermally throttled machine, stop the run and fix cooling before
  treating the numbers as a curve.
