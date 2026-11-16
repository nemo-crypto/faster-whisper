import numpy as np
import pytest

from faster_whisper.vad import SileroVADModel


class RecordingSession:
    def __init__(self):
        self.inputs = []

    def run(self, names, inputs):
        self.inputs.append(inputs["input"].copy())
        return np.zeros((len(inputs["input"]), 1)), inputs["h"], inputs["c"]


@pytest.mark.parametrize("windows", [1, 3])
@pytest.mark.parametrize("readonly", [False, True])
def test_vad_does_not_overwrite_input_or_final_window(windows, readonly):
    audio = np.arange(windows * 512, dtype=np.float32) + 1
    expected = audio.copy()
    if readonly:
        audio.flags.writeable = False
    model = SileroVADModel.__new__(SileroVADModel)
    model.session = RecordingSession()
    result = model(audio)
    np.testing.assert_array_equal(audio, expected)
    inputs = model.session.inputs[0]
    np.testing.assert_array_equal(inputs[:, 64:], expected.reshape(-1, 512))
    np.testing.assert_array_equal(inputs[0, :64], np.zeros(64))
    for i in range(1, windows):
        np.testing.assert_array_equal(inputs[i, :64], expected[i * 512 - 64 : i * 512])
    assert len(result) == windows
