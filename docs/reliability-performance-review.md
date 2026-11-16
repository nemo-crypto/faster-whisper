# Reliability and performance review — 2026-11-16

## Changes

- Include the earlier batched-generator `finally` cleanup, so explicit closure and inference errors release progress output and reset cross-request word-timestamp state.
- Reject non-positive/non-integer batch sizes before decoding audio or accessing a model. Previously negative sizes could yield no segments and zero failed later in `range`.
- Reject invalid language-detection segment counts before preprocessing/inference, with an actionable parameter-specific `ValueError`.
- Reject negative padding lengths.
- Replace indexed gathering for spectrogram truncation with slicing followed by an explicit copy. Shape, dtype, negative axes, non-contiguous input and independent-output memory semantics are preserved. Padding behavior is unchanged.

## Validation

Python 3.12, macOS arm64, CPU execution: `pytest -q tests/` passes all 43 tests, including existing real-model transcription, VAD, stereo, multilingual and tokenizer tests. Whole-repository Black, isort and Flake8 checks pass. Nine new invalid-input cases fail before the fixes; padding parity controls pass before and after.

## Reproducible microbenchmark

```python
import timeit
import numpy as np
from faster_whisper.audio import pad_or_trim
x = np.ones((80, 6000), dtype=np.float32)
print(timeit.repeat(lambda: pad_or_trim(x, 3000), number=1000, repeat=5))
```

On Apple M5, median time for 1,000 trims: 0.4758 seconds before, 0.01203 seconds after (about 39.6x for this operation). Both implementations return independent arrays. This is not an end-to-end transcription speed claim; model inference usually dominates.

## Scope and limits

No model weights, tokenizer behavior, sampling defaults, CUDA settings, dependencies, or output formats changed. Validation protects these library entry points; server authentication, upload/duration limits, rate limits and request isolation remain the responsibility of the hosting service. GPU execution, concurrent consumption of one pipeline, third-party model trust and a complete dependency vulnerability audit are outside this CPU/code review.

## Follow-up: STFT option correctness

Honor `onesided=False` for real input with a full FFT, recognize complex windows when choosing the transform, and construct the implicit rectangular window when `win_length < n_fft`. Complex input/window with an explicitly one-sided transform raises a clear ValueError. The standard real-input, real-window, one-sided feature extraction path retains its existing transform.

Nine regression cases compare short/full windows and real/complex input against explicitly framed NumPy FFT results, plus rejection of a one-sided complex window. Before the fix, six failed and three passed. This is a numerical-correctness change, with no new speedup claim.

Full suite after the follow-up: **52 passed**. Black, isort, and Flake8 checks passed.
