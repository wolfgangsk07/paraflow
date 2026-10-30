# Initial build check

- Compiler: system GCC/G++; CUDA disabled; scheduler copies = 8.
- All compilation targets reached the final executable link.
- Result: FAIL, undefined reference to `ggml_backend_cuda_host_buffer_type`.
- No engine source was changed to hide the CUDA dependency.
- CPU-only execution is not supported by this preparation snapshot as tested.
- CUDA clean-build validation and model correctness QA remain pending.
- An earlier attempt using a Conda compiler failed on its older Linux headers (`PR_SET_PTRACER`); the system compiler resolved that environment issue.

This check does not reproduce historical CUDA performance results.
