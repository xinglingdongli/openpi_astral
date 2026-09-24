"""Guided real-time chunking for OpenPI flow-matching samplers.

The prefix mask and velocity correction follow LeRobot's RTCProcessor.  The
prefix is in the model's normalized action space; callers must apply the same
input transforms used for training before passing it here.
"""

import math

import jax
import jax.numpy as jnp
import torch


# LeRobot RTCAttentionSchedule: ZEROS, ONES, LINEAR, EXP.
SCHEDULES = {"zeros": 0, "ones": 1, "linear": 2, "exp": 3}


def _jax_weights(length, delay, valid, schedule):
    indices = jnp.arange(length)
    end = jnp.minimum(valid, length)
    start = jnp.minimum(delay, end)
    linear = jnp.clip((end - indices) / jnp.maximum(end - start + 1, 1), 0.0, 1.0)
    linear = jnp.where(indices < start, 1.0, linear)
    linear = jnp.where(indices >= end, 0.0, linear)
    exp = linear * jnp.expm1(linear) / math.expm1(1.0)
    return jnp.where(
        schedule == 0, indices < start,
        jnp.where(schedule == 1, indices < end, jnp.where(schedule == 3, exp, linear)),
    ).astype(jnp.float32)


def _torch_weights(length, delay, valid, schedule, device):
    indices = torch.arange(length, device=device)
    end = min(valid, length)
    start = min(delay, end)
    linear = ((end - indices).float() / max(end - start + 1, 1)).clamp(0, 1)
    linear = torch.where(indices < start, 1.0, linear)
    linear = torch.where(indices >= end, 0.0, linear)
    if schedule == 0:
        return (indices < start).float()
    if schedule == 1:
        return (indices < end).float()
    if schedule == 3:
        return linear * torch.expm1(linear) / math.expm1(1.0)
    return linear


def _guidance_weight(time, maximum):
    # LeRobot uses tau=1-time; OpenPI and LeRobot both sample from t=1 to 0.
    tau = 1.0 - time
    if isinstance(time, torch.Tensor):
        safe_tau = tau.clamp_min(1e-6)
        safe_time = time.clamp_min(1e-6)
        return ((time * time + tau * tau) / (safe_time * safe_tau)).clamp(max=maximum)
    safe_tau = jnp.maximum(tau, 1e-6)
    safe_time = jnp.maximum(time, 1e-6)
    return jnp.minimum((time * time + tau * tau) / (safe_time * safe_tau), maximum)


def guide_jax(x_t, time, prefix, delay, valid, maximum, schedule, denoise_fn):
    """Apply LeRobot's exact JAX VJP guidance to one denoising step.

    LeRobot computes ``grad(x1_t, x_t, grad_outputs=error)``.  This is a
    Jacobian-vector product, not simply ``error``; ``denoise_fn`` is therefore
    passed in so JAX can differentiate through the model velocity.
    """
    weights = _jax_weights(x_t.shape[1], delay, valid, schedule)[None, :, None]
    prefix = jnp.pad(
        prefix.astype(x_t.dtype),
        ((0, 0), (0, x_t.shape[1] - prefix.shape[1]), (0, 0)),
    )

    def x1_fn(x):
        v = denoise_fn(x)
        return x - time * v

    x1_t, pullback = jax.vjp(x1_fn, x_t)
    error = (prefix - x1_t) * weights
    correction = pullback(error)[0]
    v_t = (x_t - x1_t) / jnp.maximum(time, 1e-6)
    return v_t - _guidance_weight(time, maximum) * correction


def guide_torch(x_t, time, prefix, delay, valid, maximum, schedule, denoise_fn):
    """Apply LeRobot's exact PyTorch autograd guidance."""
    weights = _torch_weights(x_t.shape[1], delay, valid, schedule, x_t.device)[None, :, None]
    prefix = torch.nn.functional.pad(
        prefix.to(dtype=x_t.dtype, device=x_t.device), (0, 0, 0, x_t.shape[1] - prefix.shape[1])
    )
    v_t = denoise_fn(x_t)
    x1_t = x_t - time * v_t
    error = (prefix - x1_t) * weights
    correction = torch.autograd.grad(
        x1_t,
        x_t,
        grad_outputs=error.detach(),
        retain_graph=False,
        create_graph=False,
    )[0]
    return v_t - _guidance_weight(time, maximum) * correction
