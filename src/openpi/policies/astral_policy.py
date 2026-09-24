"""OpenPI input/output transforms for the Astral single-arm teleop robot (loopkok astral_ws).

Dataset conventions (produced by astral_data_collect -> LeRobot v2.1 conversion):
  - observation.state: 8-dim = left arm 7 joints (rad) + left gripper close ratio (0..1)
  - action:            8-dim, next_state semantics (action[t] == state[t+1]),
                       absolute joint angles (NOT deltas)
  - observation.images.base: D435i head-mounted third-person 1080p
  - observation.images.left_wrist: left wrist USB 720p
  - observation.images.right_wrist: right wrist USB (new 3-camera data; old
    2-camera episodes don't have it and use the 2-slot config, see below)

Camera slot mapping (pi05 has three fixed slots): base -> base_0_rgb,
left_wrist -> left_wrist_0_rgb, right_wrist_0_rgb zero-padded with mask=False
when the 2-slot config (pi05_astral / pi05_astral_lora) is used; the
*_3cam configs map it to a real right_wrist dataset key.

Input key convention (shared by training and inference; the dataset's native keys
are mapped onto these by the repack transform in LeRobotAstralDataConfig):
  - "observation/state"       <- observation.state
  - "observation/image"       <- observation.images.base (head third-person)
  - "observation/wrist_image" <- observation.images.left_wrist (left wrist)
  - "actions"                 <- action (training only)
  - "prompt"                  <- prompt (injected from LeRobot task; empty falls
                                 back to default_prompt)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_astral_example() -> dict:
    """Creates a random input example for smoke-testing the policy server."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AstralInputs(transforms.DataTransformFn):
    """Converts Astral dataset / inference inputs to the pi05 model input format.

    Used for both training and inference. The camera wiring is fully driven by
    ``cameras`` (model slot -> input key). Slots absent from the mapping, or
    configured slots omitted by an inference request, are zero-padded with
    mask=False. Training still requires all configured dataset camera keys.
    """

    # Determines which model is used (image slot / mask conventions). Do not change.
    model_type: _model.ModelType
    # Model image slot -> key in the (post-repack) input dict. Slots that are
    # not listed are zero-padded with mask=False.
    cameras: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "base_0_rgb": "observation/image",
            "left_wrist_0_rgb": "observation/wrist_image",
        }
    )
    # Fallback prompt used when the environment passes none (or an empty string).
    # None passes prompts through unchanged.
    default_prompt: str | None = None

    def __call__(self, data: dict) -> dict:
        # pi05/pi0 convention: missing cameras are zero-padded with mask=False
        # (PI0_FAST does not mask padding images, but this robot is trained as
        # pi05; fail loudly for anything else).
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"AstralInputs only supports pi0/pi05, got: {self.model_type}")

        all_slots = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        # Inference may deliberately omit a configured camera for an ablation.
        # Presence, rather than the static training camera map, determines the
        # runtime mask. Dataset training is unchanged because repack supplies all
        # configured keys and still fails earlier if a dataset feature is absent.
        parsed = {
            slot: _parse_image(data[key])
            for slot, key in self.cameras.items()
            if key in data
        }
        if not parsed:
            raise ValueError("AstralInputs requires at least one camera image")
        # Reference image for zero-padding shapes: any provided camera.
        ref = next(iter(parsed.values()))
        images = {slot: parsed.get(slot, np.zeros_like(ref)) for slot in all_slots}
        image_masks = {slot: slot in parsed for slot in all_slots}

        inputs = {
            "state": data["observation/state"],
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # Empty prompt (early episodes recorded with an empty task string) is
        # dropped so that the downstream InjectDefaultPrompt can kick in.
        prompt = data.get("prompt")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if prompt:
            inputs["prompt"] = prompt
        elif self.default_prompt:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class AstralOutputs(transforms.DataTransformFn):
    """Trims model outputs back to the dataset action dimension (inference only).

    The model action_dim is 32; only the first ``action_dim`` dims are real for
    this robot, the rest is padding. The value is derived from the dataset meta
    by LeRobotAstralDataConfig (8 = single arm 7 joints + gripper).
    """

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
