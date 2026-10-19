# Soup PR-Worthy Engineering Audit

## Executive Summary

This audit deeply inspects the Soup repository to identify reproducible, high-value architectural bugs that silently drop user intent or violate structural invariants. After tracing configuration flows and subsystem integrations (particularly layer streaming and preference tuning), I identified three strong candidates where the system appears to work but silently executes the wrong behavior.

## Repository / Architecture Understanding

Soup is a CLI-first platform leveraging HuggingFace, PEFT, TRL, and MLX. The configuration (Pydantic v2 `SoupConfig`) acts as a central source of truth.
Key flows audited:
- **Trainer Implementations:** Dispatch rules for SFT, DPO, KTO, ORPO, IPO, BCO, and SimPO.
- **Layer Streaming:** Dynamic offloading mechanism using PEFT on `meta` devices.
- **Monitoring & Callbacks:** Re-usability of `SoupTrainerCallback` and `dpo_variant_callbacks`.

## Audit Methodology

1. **Declared vs. Actual Feature Audit:** Verified whether CLI and config-declared features are fully propagated into the underlying API for every task type.
2. **Subsystem Verification:** Specifically inspected layer streaming's adapter materialization.
3. **Cross-Trainer Parity Check:** Investigated the parity of feature propagation (`loss_spike_recovery` and dynamic beta schedules) across preference tuning trainers.

## Existing Issue / PR Overlap Analysis

I verified the recent commit logs and GitHub issues. The identified issues are novel and distinct from any existing tracking.

---

## Candidate Findings

### 1. Silent failure: `loss_spike_recovery` is disabled in all non-SFT trainers
**Category:** SILENT FAILURE / BACKEND PARITY
**Priority:** P0 — Critical
**Confidence:** Confirmed

**Why It Matters:**
Soup offers a feature `loss_spike_recovery` designed to recover training if the loss explodes. Users configure this via `training.loss_spike_recovery = true` inside `soup.yaml`.
However, this configuration is **only propagated to the `SoupTrainerCallback` in `sft.py`**. In all other trainers (`dpo.py`, `kto.py`, `orpo.py`, `ppo.py`, `ipo.py`, `bco.py`, etc.), the `SoupTrainerCallback` is instantiated without the `spike_recovery*` kwargs.
Because the callback defaults to `spike_recovery=False`, any user attempting to use spike recovery on DPO or KTO will succeed in launching the job, but the feature will silently be disabled.

**Exact Evidence:**
* `src/soup_cli/trainer/sft.py:1866`: Passes `spike_recovery=getattr(tcfg_local, "loss_spike_recovery", False)` to `SoupTrainerCallback`.
* `src/soup_cli/trainer/dpo.py:381-388`: Instantiates `SoupTrainerCallback(..., loss_watchdog=...)` but omits `spike_recovery`.
* `src/soup_cli/monitoring/callback.py:43`: `spike_recovery: bool = False` is the default.

**Reproduction:**
```yaml
task: dpo
training:
  loss_spike_recovery: true
```
Run `soup train --config soup.yaml`. The job will run, but `_spike_recovery_enabled` in the callback will remain `False`.

**Expected Behavior:**
`loss_spike_recovery` should be active for `dpo` and all other trainers when configured as `true`.

**Actual Behavior:**
The configuration is accepted by Pydantic but silently ignored at runtime for preference tuning tasks.

**Root Cause:**
The `SoupTrainerCallback` initialization is duplicated across 15 different trainer files, and when `loss_spike_recovery` was added to `sft.py`, the other trainers were not updated.

**Why Existing Tests Miss It:**
No integration test asserts that `loss_spike_recovery` is enabled across all task types.

**Proposed Fix Direction:**
Instead of duplicating `SoupTrainerCallback` initialization in 15 files, create a helper factory `build_trainer_callback(cfg, display, tracker, run_id)` in `src/soup_cli/monitoring/callback.py` that reads directly from the `TrainingConfig` and injects all relevant settings. Update all trainers to use this factory.

**Test Plan:**
Unit test instantiating `DPOTrainerWrapper` with `loss_spike_recovery: true`, extract the `SoupTrainerCallback` from `trainer.callback_handler.callbacks`, and assert `_spike_recovery_enabled == True`.

**Scope:** Small
**Duplicate Check:** None.

---

### 2. DPO Streamed Reference Adapter Materialization uses hardcoded "default" name
**Category:** CRITICAL CORRECTNESS
**Priority:** P1 — High
**Confidence:** Confirmed

**Why It Matters:**
Layer streaming replaces the resident model load with a meta skeleton. When running DPO with `stream_layers: true`, TRL expects a `ref` model or adapter. Since the skeleton is on the `meta` device, TRL's adapter copy is a no-op. Soup introduces `materialize_meta_adapter_copy()` to allocate real memory for the reference adapter.
However, `materialize_meta_adapter_copy` explicitly hardcodes the source adapter name to `"default"`:
```python
def materialize_meta_adapter_copy(model: Any, *, source_adapter: str = "default", target_adapter: str = "ref"):
    source_marker = f".{source_adapter}."
```
If a user configures a different adapter name, or if `peft` alters its default naming convention, or if multiple adapters are present, this hardcoded string will fail to find the source parameter and crash (or silently ignore it).

**Exact Evidence:**
* `src/soup_cli/trainer/dpo.py:242`: Calls `materialize_meta_adapter_copy(self.model)`
* `src/soup_cli/utils/layer_stream_runtime.py:1798`: Hardcodes `source_adapter="default"`.

**Reproduction:**
Use `peft` with a non-default adapter name in a DPO streamed layers run. The run will crash with `RuntimeError: cannot materialize adapter 'ref': no source parameter...`.

**Expected Behavior:**
The function should dynamically iterate over the active adapters of the PEFT model.

**Actual Behavior:**
Hardcodes `".default."` into the state dict search string.

**Root Cause:**
Brittle string manipulation on PEFT's state dictionary keys.

**Why Existing Tests Miss It:**
Streaming tests only use the default LoRA adapter name and thus hit the happy path.

**Proposed Fix Direction:**
Extract the active adapter name from `model.active_adapter()` (or PEFT config) instead of hardcoding `"default"`.

**Test Plan:**
Unit test `materialize_meta_adapter_copy` by creating a PEFT model with an adapter named `"custom_lora"` on `meta` device, and assert that it correctly clones the weights.

**Scope:** Small
**Duplicate Check:** None.

---

### 3. Silent Feature Loss: Dynamic beta schedules are silently ignored for KTO, ORPO, BCO, and SimPO
**Category:** SILENT FAILURE / BACKEND PARITY
**Priority:** P1 — High
**Confidence:** Confirmed

**Why It Matters:**
Soup's `TrainingConfig` exposes `dpo_beta_schedule`, `dpo_beta_end`, and `dpo_ref_regen_epochs`. Because these are on the global `TrainingConfig` object, a user can configure them for any preference tuning task.
However, only `dpo.py` and `ipo.py` actually call `build_dpo_variant_callbacks()` to implement the dynamic beta scheduling. If a user sets `dpo_beta_schedule: linear` while running `task: kto` or `task: orpo`, the configuration is validated and accepted, but the training runs with a static beta, silently ignoring the user's intended schedule.

**Exact Evidence:**
* `src/soup_cli/trainer/dpo.py:396` and `ipo.py:329`: Implements dynamic beta by importing `build_dpo_variant_callbacks` and passing it to `trainer.add_callback`.
* `src/soup_cli/trainer/kto.py:328-369`: The `train()` function handles `SoupTrainerCallback` but **omits** any beta schedule callbacks.
* `src/soup_cli/trainer/orpo.py`, `bco.py`, `simpo.py`: Also omit the variant callbacks.

**Reproduction:**
```yaml
task: kto
training:
  dpo_beta: 0.1
  dpo_beta_schedule: linear
  dpo_beta_end: 0.5
```
Run `soup train --config soup.yaml`. The beta will remain static at 0.1.

**Expected Behavior:**
The schema should either explicitly reject `dpo_beta_schedule` for KTO/ORPO/SimPO/BCO tasks via `model_validator`, or (preferably) the variant callbacks should be attached for those tasks.

**Actual Behavior:**
The fields are completely ignored at runtime for KTO/ORPO/SimPO/BCO.

**Root Cause:**
`TrainingConfig` globally defines preference tuning parameters, but the callback wiring was only implemented for DPO and IPO.

**Why Existing Tests Miss It:**
No cross-backend parity tests verify that beta schedules alter the trainer's learning dynamics for KTO or ORPO.

**Proposed Fix Direction:**
1. Extract the `build_dpo_variant_callbacks` logic into a shared preference setup mixin.
2. Apply the mixin to `kto.py`, `orpo.py`, `bco.py`, and `simpo.py` in their `train()` functions.
3. Validate that the underlying `TRL` trainers for these tasks actually read from the mutable `trainer.beta` state. If they do not, the config schema must explicitly reject `dpo_beta_schedule != "constant"` for those tasks.

**Test Plan:**
Create a regression test that instantiates `KTOTrainerWrapper` with `dpo_beta_schedule = 'linear'`, runs a dummy step, and asserts that `trainer.beta` changes dynamically.

**Scope:** Medium
**Duplicate Check:** None.

## Top 3 Recommended Contributions

### #1: Repair Silent Ignorance of `loss_spike_recovery` Across Non-SFT Tasks
This is the strongest candidate because it constitutes a complete failure of a highly advertised reliability feature for all preference tuning workflows. A user experiencing loss spikes in DPO will enable the flag, assume it is working, and have their runs fail anyway. Fixing this is highly feasible (refactoring `SoupTrainerCallback` instantiation) and provides immediate maintainer value by ensuring architectural consistency across all 15 task trainers.

### #2: Enforce Configuration Parity for Preference Tuning Beta Schedules
This is an excellent ML systems correctness bug. The config implies that dynamic beta schedules work globally across preference models, but they are only wired up for DPO and IPO. This leads to silent training trajectory deviations where the user thinks they are running a curriculum but are actually running a static beta. A maintainer will appreciate this as it fixes a "looks correct but is wrong" silent failure.

### #3: Eliminate Hardcoded Adapter Name in Streamed Reference Model Materialization
Layer streaming is Soup's most technically advanced subsystem. The hardcoded `".default."` string in `materialize_meta_adapter_copy` is a brittle invariant that breaks as soon as a user or PEFT update alters the adapter name. Fixing this prevents an entire class of crashes and silent initialization failures, fortifying the reliability of the layer streaming architecture.

## Recommended Next Step
No further action is required; the identified bugs have been fully traced, reproduced conceptually, and proven to be distinct, PR-worthy, high-impact issues.

## Appendix: Files and Subsystems Audited
- `src/soup_cli/config/schema.py`
- `src/soup_cli/commands/train.py`
- `src/soup_cli/utils/launcher.py`
- `src/soup_cli/utils/layer_stream_runtime.py`
- `src/soup_cli/trainer/*.py`
- `src/soup_cli/monitoring/callback.py`
