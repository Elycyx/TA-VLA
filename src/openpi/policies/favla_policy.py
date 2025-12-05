import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FavlaInputs(transforms.DataTransformFn):
    """Inputs for the Favla policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - force: [history, 7]
    - actions: [action_horizon, 7]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        # make state all 0
        state = np.zeros_like(state)

        in_images = data["images"]

        # Assume that base image always exists.
        base_image = _parse_image(in_images["images"])

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": np.zeros_like(base_image),
            # Pad any non-existent images with zero-arrays of the appropriate shape.
            "right_wrist_0_rgb": np.zeros_like(base_image)
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.False_,
            # Mask any non-existent images with False (if ``mask_padding`` is True).
            "right_wrist_0_rgb": np.False_,
        }


        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "effort" in data:
            inputs["effort"] = data["effort"]

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FavlaOutputs(transforms.DataTransformFn):
    """Outputs for the Favla policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        actions = np.asarray(data["actions"][:, :7])
        return {"actions": actions}
