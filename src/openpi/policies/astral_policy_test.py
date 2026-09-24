import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import astral_policy


def _transform() -> astral_policy.AstralInputs:
    return astral_policy.AstralInputs(
        model_type=_model.ModelType.PI05,
        cameras={
            "base_0_rgb": "observation/camera/base_0_rgb",
            "left_wrist_0_rgb": "observation/camera/left_wrist_0_rgb",
        },
    )


def test_missing_configured_wrist_is_zero_padded_and_masked_out() -> None:
    base = np.full((32, 48, 3), 123, dtype=np.uint8)
    result = _transform()({
        "observation/state": np.zeros(8),
        "observation/camera/base_0_rgb": base,
        "prompt": "pick up",
    })

    assert np.array_equal(result["image"]["base_0_rgb"], base)
    assert result["image_mask"]["base_0_rgb"]
    assert not result["image_mask"]["left_wrist_0_rgb"]
    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert not result["image"]["left_wrist_0_rgb"].any()
    assert result["image"]["left_wrist_0_rgb"].shape == base.shape


def test_all_configured_cameras_remain_valid() -> None:
    base = np.full((24, 24, 3), 50, dtype=np.uint8)
    wrist = np.full((24, 24, 3), 200, dtype=np.uint8)
    result = _transform()({
        "observation/state": np.zeros(8),
        "observation/camera/base_0_rgb": base,
        "observation/camera/left_wrist_0_rgb": wrist,
    })

    assert result["image_mask"]["base_0_rgb"]
    assert result["image_mask"]["left_wrist_0_rgb"]
    assert np.array_equal(result["image"]["left_wrist_0_rgb"], wrist)


def test_rejects_request_without_any_camera() -> None:
    with pytest.raises(ValueError, match="at least one camera"):
        _transform()({"observation/state": np.zeros(8)})
