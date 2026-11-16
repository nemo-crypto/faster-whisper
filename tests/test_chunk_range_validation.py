import numpy as np
import pytest

from faster_whisper.vad import collect_chunks


@pytest.mark.parametrize(
    "start,end",
    [
        (-1, 4),
        (0, 11),
        (8, 4),
        (11, 12),
        (0.5, 4),
        (0, 4.5),
        (False, 4),
        (0, True),
    ],
)
def test_invalid_ranges_are_rejected_before_slicing(start, end):
    audio = np.arange(10, dtype=np.float32)
    with pytest.raises(ValueError, match="chunk.*range"):
        collect_chunks(audio, [{"start": start, "end": end}], sampling_rate=10)


def test_valid_numpy_integer_ranges_match_metadata():
    audio = np.arange(10, dtype=np.float32)
    chunks = [
        {"start": np.int64(1), "end": np.int64(4)},
        {"start": np.int64(6), "end": np.int64(10)},
    ]
    outputs, metadata = collect_chunks(
        audio, chunks, sampling_rate=10, max_duration=0.5
    )
    assert len(outputs) == 2
    for output, info in zip(outputs, metadata):
        assert len(output) / 10 == info["duration"]
    np.testing.assert_array_equal(outputs[0], audio[1:4])
    np.testing.assert_array_equal(outputs[1], audio[6:10])
    assert metadata[1]["offset"] == 0.3


def test_zero_length_chunk_keeps_existing_empty_output():
    outputs, metadata = collect_chunks(
        np.zeros(10, dtype=np.float32), [{"start": 10, "end": 10}], sampling_rate=10
    )
    assert outputs[0].size == 0
    assert metadata[0]["duration"] == 0
