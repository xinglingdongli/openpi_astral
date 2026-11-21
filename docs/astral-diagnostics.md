# Optional Astral inference observations

`Policy(..., diagnostic_hook=None)` preserves the existing default. An observer can be enabled or disabled later with `policy.set_diagnostic_hook(hook)` / `policy.set_diagnostic_hook(None)`.

The hook receives `(event, data)`: `model_input` is the CPU input-transform snapshot before batching/device transfer; `model_host_output` is the model output after the existing CPU conversion, before output transforms. Input images are still uint8; the existing Observation conversion subsequently maps them to [-1, 1]. State normalization, masks, prompt tokens and RTC prefix transforms remain unchanged.

Snapshots are owned CPU NumPy data. Hook exceptions do not escape and callbacks cannot mutate inference inputs. A callable may expose `enabled=False` to suspend snapshots without reconfiguring the policy. Device tensors are not copied or synchronized for diagnostics. There are no new model arguments, RNG changes, output fields, kernel timing claims or default logging destinations.

Astral's optional `serve.py --diagnostics-dir ...` installs this hook and writes snapshots through its bounded background recorder. The existing `PolicyRecorder` and native server remain unchanged. Target GPU model execution still requires its own validation; the lightweight observer test does not require CUDA/Flax/Torch.
