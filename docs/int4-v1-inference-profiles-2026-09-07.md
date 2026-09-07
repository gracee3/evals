# INT4 v1 inference presets

The INT4-only acceptance suite uses the validated single-GPU preset:

- physical GPU0 by UUID
- TP1
- 98,304-token maximum context
- FP8 `float8_e4m3fn` KV cache
- 3.5 GiB KV reservation
- one sequence, prefix caching, non-eager execution, and 2,048-token chunked prefill

This profile completed a 95,998-token prompt plus 64 generated tokens. Cold
prefill was approximately 813 tok/s at the near-window probe and short decode
was 45.4 tok/s. The measured run retained 1.91 GiB on GPU0 after the request.

The quantization repository also records a TP2 comparison using the same 96K
and 3.5 GiB-per-GPU FP8 settings. That run reached 68.4 tok/s short decode and
67.7 tok/s after a 96K prompt, with 10.49 GiB free on each GPU. It is a separate
dual-GPU serving preset and is not the default for this single-model acceptance
suite.

Raw evidence remains outside Git under `/data/qwen38-int8-lab/int4-v1/`.
