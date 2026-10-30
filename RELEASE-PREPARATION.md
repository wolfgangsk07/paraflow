# Private release preparation

Project name: **ParaFlow** (Chinese: **并流**).

Repository visibility must remain **private** until the owner explicitly authorizes publication. Do not create a public release, public Pages site, or public benchmark announcement during preparation.

Initial source target: the V72b component stack. Preserve provenance and licenses. Unfinished experimental changes must remain separate.

Before public release:

- Establish the exact upstream revision and complete attribution.
- Reproduce a clean CUDA build from this source tree; document supported hardware.
- Re-run correctness checks against the resulting binary, with explicit treatment of numerical differences.
- Validate automatic allocation and request lifecycle behavior, cache and priority paths, memory pressure and recovery.
- Separate same-hardware engine gains from added-hardware and parameter gains.
- Publish only sanitized benchmark inputs or reproducible synthetic workloads and aggregated results.

Current repository contents are preparation material. No production deployment is implied.
