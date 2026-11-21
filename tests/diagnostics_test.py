import numpy as np

from openpi.policies.diagnostics import notify


def test_observer_cannot_mutate_inputs_and_failures_do_not_escape():
    original = np.array([1.0, 2.0])

    def hook(event, data):
        data["state"][:] = 99
        raise OSError("recording failed")

    notify(hook, "model_input", {"state": original})
    np.testing.assert_array_equal(original, [1.0, 2.0])
