> Historical design documentation inherited from the base snapshot. It is not the V72b release specification or a validated launch recipe. See README.md for current preparation status.

# Layer pipeline, MTP, VIP and dynamic KV

This fork retains layer partitioning across GPUs. A request visits every target
model stage. Independent lanes can submit target verification and ordinary
decode work while earlier work is still running. Completed work is selected by
a decode ticket containing its own logits and MTP hidden states. Sampling a
selected ticket waits for its output completion events rather than synchronizing
the whole target context.
Text prefill uses the same per-sequence chunk boundaries in standalone and fused
cars. Whole chunks are admitted with generation capacity reserved, keeping
recurrent scan boundaries independent of the other active requests. A loaded
multimodal projector does not exclude text-only chat requests from this path;
only prompts containing actual media chunks are excluded from text fusion.

For MTP, each lane has a draft-head replica on its corresponding GPU. Draft
jobs run independently; the main server thread owns verification, sampling,
request lifecycle and output delivery. The number of lanes defaults to the
number of model stages. Request counts need not be multiples of the lane count.
Mixed MTP and ordinary traffic is supported; unequal workloads can still leave
idle intervals. The scheduler does not promise zero idle time for arbitrary
request lengths or draft acceptance rates.

## Starting a server

Use the regular target model plus an MTP-compatible sidecar. For a grafted head,
token IDs, target embeddings and output projection must agree with the target
model. A sidecar containing the original target embedding/output tensors and
only the donor MTP tensors avoids duplicating the backbone.

```sh
llama-server -m target.gguf -md target-heads-and-mtp.gguf \
  -ngl 99 -ts 1,1,1 -np 15 -c 150000 -b 16384 -ub 1024 -fa on \
  --pipe-lanes 3 --pipe-fuse --pipe-prefill-chunk 768 --pipe-pf-inflight 4 \
  -kvd --kv-dynamic-seq-max 65536 --kv-spill --vip \
  --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-ngl 99
```

`-kvd` reserves independent virtual KV address spaces for sequences, backed by
one fixed shared CUDA physical-page pool. `-c` limits the shared token budget;
`--kv-dynamic-seq-max` limits one sequence and must not exceed `-c`. This mode
requires CUDA KV offload, Flash Attention and non-unified KV. It is distinct
from `-kvu`; unified KV disables the layer pipeline.

`--kv-spill` can suspend a request to host RAM when active KV approaches the
shared limit. Both target and draft state are restored before that request
resumes. This needs sufficient host memory; it does not increase the configured
shared physical GPU pool. The allocator reserves CUDA pages in advance, so
small pools have page-granularity overhead.
Capacity accounting counts occupied KV cells individually. Image patches that
share a temporal position still consume separate cells in this budget.

## Independent request choices

The native completion and OpenAI chat endpoints accept these extension fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `mtp` | `false` | Enable the configured MTP head for this request. |
| `priority` | `0` | A positive value selects VIP treatment when `--vip` is enabled. |
| `speculative.n_max` | Startup limit | Maximum draft length, bounded by the startup limit; zero disables drafting. |

All four priority/MTP combinations are valid. Ordinary requests with `mtp:false`
do not execute draft prefill, draft catch-up or proposal jobs. VIP does not
implicitly enable MTP.

```json
{
  "messages": [{"role": "user", "content": "Explain this calculation."}],
  "mtp": true,
  "priority": 1,
  "speculative.n_max": 3,
  "seed": 731,
  "stream": true
}
```

A VIP request suspends ordinary execution down to
`max(0, lanes - active_VIP)`. If all server slots are occupied, it can borrow a
paused slot. The saved continuation includes sampler RNG, token and SSE cursor,
target/draft KV and MTP pending verification state. Disconnecting a VIP releases
its borrowed slot and allows the saved request to continue. Several VIP requests
and capacity paging use generation ownership guards to avoid restoring a
continuation into a reused slot.

## Output and correctness

MTP uses target sample-and-match verification. Draft sampling does not consume
the target sampler's RNG. Rejected draft tokens never enter the response.
Each committed token ID is emitted exactly once, including byte-fallback tokens
whose UTF-8 text is temporarily incomplete. Such events can have an empty text
fragment; subsequent events complete the character. Streamed and non-streamed
text use the same valid UTF-8 prefix at a token limit.

For the validated Volta Q8 target/Q6 MTP configuration, stable Q8 matrix-vector
warps, canonical recurrent prefill, verification microbatching and canonical
small-query Flash Attention keep the numerical path stable across ordinary and
MTP batches. Recurrent sequence compaction preserves every pending rollback
plane. These changes are required for strict seeded token comparisons; model
quality or a similar-looking answer is not a substitute for token equality.

`LLAMA_VERIFY_CONTINUATION_ISOLATION=1` enables expensive validation that
suspending/restoring one sequence does not change the serialized states of
other sequences. It is intended for pressure and continuation acceptance.

Tests must compare against the same target weights and sampling configuration.
Useful acceptance cases include MTP lengths 1/3/5, mixed traffic, long seeded
outputs, prefill during decode, VIP disconnection, slot reuse, physical-pool
pressure, cached ordinary/MTP transitions and truncated UTF-8 characters.
Hardware acceptance here covers three V100 GPUs. Other GPU counts use the same
stage-count-based scheduling, but need their own hardware validation.
