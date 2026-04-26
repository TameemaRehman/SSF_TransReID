# SSF-on-TransReID — Implementation Notes

This document describes how **Scale-Shift Fine-Tuning (SSF)** from the NeurIPS 2022 SSF paper is integrated into the **TransReID** Vision Transformer backbone in this repository, which files implement each part, and **why** those design choices were made.

## Motivation

- **TransReID** is a transformer-based person re-identification model. Full fine-tuning updates all backbone weights, which is data-hungry and expensive.
- **SSF** adds a small set of per-channel scale (`gamma`) and shift (`beta`) parameters applied to activations. With backbone weights frozen, only SSF (and the usual Re-ID head: classifier, bottleneck, etc.) trains—**parameter-efficient fine-tuning (PEFT)** aligned with the SSF methodology.
- The integration targets **faithfulness to the SSF paper** where noted in code (four adaptation points per block, optional block subset, higher LR and no weight decay on SSF parameters, optional full backbone freeze).

## What SSF Does (Core Module)

For a tensor whose last dimension is channel size `C`, SSF computes:

\[
\text{SSF}(x) = x \odot \gamma + \beta
\]

with `gamma` initialized to **ones** and `beta` to **zeros**, so the network starts in the **identity** regime and pretrained behavior is preserved until optimization moves the SSF parameters.

**File:** `model/peft/ssf.py`

| Piece | Purpose |
|--------|---------|
| `SSF` | `nn.Module` holding `gamma` / `beta`; forward supports `(…, C)` layouts used in ViT tokens. |
| `merge_ssf_into_linear` | Optional utility to fold an SSF applied after a `nn.Linear` into that layer’s weights (in-place); resets SSF to identity. |
| `unmerge_ssf_from_linear` | Restores weights and returns a new `SSF` with stored `gamma`/`beta`. |

**File:** `model/peft/__init__.py` — re-exports `SSF`, `merge_ssf_into_linear`, `unmerge_ssf_from_linear` for a clean PEFT package surface.

## Where SSF Is Attached in TransReID (ViT)

**File:** `model/backbones/vit_pytorch.py`

### 1. Per-transformer-block SSF (“SSF-ADA”)

**Class:** `Block`

When `ssf_enabled=True`, each block adds **four** `SSF` modules, inserted after the operations called out in comments as matching **SSF paper Table 6b** (all four adaptation points):

| Module | Position in forward | Rationale |
|--------|----------------------|-----------|
| `ssf_norm1` | After `LayerNorm` on the residual stream, **before** attention | Adapts normalized features feeding attention. |
| `ssf_attn` | After attention output (post projection / dropout), **before** residual add | Adapts attention outputs. |
| `ssf_norm2` | After second `LayerNorm`, **before** MLP | Adapts features feeding the MLP. |
| `ssf_mlp` | After MLP (post dropout), **before** residual add | Adapts MLP outputs. |

This is stricter than only adapting after attention and MLP: it matches the paper’s full **four-operation** pattern and keeps residual structure (SSF sits on branch paths before adds).

**Selective blocks:** `TransReID.__init__` passes `ssf_enabled=ssf_enabled and (len(ssf_blocks) == 0 or i in ssf_blocks)` into each `Block`. An **empty** `ssf_blocks` tuple means **all** depth indices get SSF; otherwise only listed indices (e.g. last layers only) get SSF for fewer parameters.

### 2. Global SSF on the token sequence

When `ssf_enabled=True`, `TransReID` also defines:

| Module | Position | Rationale |
|--------|----------|-----------|
| `ssf_patch_embed` | Immediately after `patch_embed`, before CLS concat | Adapts patch tokens right after projection (analogous to adapting early representation). |
| `ssf_final_norm` | After the final `norm` (LayerNorm), before taking `x[:, 0]` (CLS) | Adapts the normalized sequence before global feature readout. |

### 3. Factory entry points

**Functions:** `vit_base_patch16_224_TransReID`, `vit_small_patch16_224_TransReID`, `deit_small_patch16_224_TransReID` — each forwards `ssf_enabled` and `ssf_blocks` into `TransReID` so configs can toggle SSF without changing call sites elsewhere.

## Wiring SSF Into the Full Re-ID Model

**File:** `model/make_model.py`

| Component | What was implemented | Why |
|-----------|----------------------|-----|
| `_freeze_non_ssf` | Sets `requires_grad=False` for every parameter whose name does **not** contain `'ssf'`. | Matches the SSF recipe: train adapters, not the pretrained backbone. |
| `build_transformer` | Passes `ssf_enabled=cfg.PEFT.SSF.ENABLED` and `ssf_blocks=cfg.PEFT.SSF.BLOCKS` into the ViT factory; optionally calls `_freeze_non_ssf(self.base.named_parameters())`. | Non-JPM TransReID path gets the same PEFT behavior. |
| `build_transformer_local` | Same factory args; freeze runs on `self.base` **before** `deepcopy` of the last block into JPM branches `b1` / `b2`; then `_freeze_non_ssf` on `b1` and `b2`. | **JPM** duplicates the last block: freezing must happen so copied branches do not accidentally leave backbone weights trainable; re-applying the rule on `b1`/`b2` covers edge cases (e.g. last block excluded from `ssf_blocks`). |

**File:** `model/__init__.py` — exposes `make_model` (unchanged surface; SSF is entirely cfg-driven).

## Configuration

**File:** `config/defaults.py` — under `PEFT.SSF`:

| Key | Default | Meaning |
|-----|---------|---------|
| `ENABLED` | `False` | Master switch for SSF in the ViT backbone. |
| `BLOCKS` | `()` | Empty = all transformer blocks; non-empty = only those indices receive per-block SSF. |
| `MERGE_ON_SAVE` | `False` | Reserved in config; **not referenced** elsewhere in Python training/save code in this repo (merge helpers exist in `model/peft/ssf.py` for manual or future use). |
| `LR` | `0.0` | `0.0` means “use optimizer rule: **10×** `SOLVER.BASE_LR` for SSF params” (see below). |
| `FREEZE_BACKBONE` | `False` | When `True` with `ENABLED`, freezes all non-SSF parameters in backbone branches as described above. |

**Example configs**

- `configs/Market/vit_transreid_ssf.yml` — SSF on Market-1501 with **SGD**, explicit `PEFT.SSF.LR`, `FREEZE_BACKBONE: True`, etc.
- `configs/Market/vit_transreid_stride_ssf_new.yml` — Variant aligned with SSF paper training notes in comments (**AdamW**, warmup, `LR: 0.0` for 10× base LR on SSF, `BLOCKS: ()` for all 12 blocks).

Adjust `DATASETS.ROOT_DIR`, `MODEL.PRETRAIN_PATH`, and `OUTPUT_DIR` for your machine.

## Optimizer Behavior for SSF Parameters

**File:** `solver/make_optimizer.py`

- Parameters whose names contain `'ssf'` get learning rate `ssf_lr = cfg.PEFT.SSF.LR` if `LR > 0`, else **`cfg.SOLVER.BASE_LR * 10`**.
- SSF parameters use **`weight_decay = 0.0`**, so scale/shift are not regularized toward zero (which would fight the adapter’s role).
- **AdamW fix:** the optimizer is constructed as `torch.optim.AdamW(params)` **without** global `lr` / `weight_decay`, so each param group’s `lr` and `weight_decay` are preserved. Passing global AdamW defaults would override per-group settings and break SSF’s custom LR and zero decay.

## Verification and Sanity Tools

| File | Role |
|------|------|
| `tests/test_ssf_integration.py` | Pytest coverage: SSF registration, selective blocks, gradients, identity init vs baseline forward, optimizer param groups, merge/unmerge helpers, etc. |
| `tools/check_ssf.py` | CLI diagnostic: load a YAML and print SSF parameter counts/names, or `--cpu-only` tiny `TransReID` forward/backward without dataset/GPU. |
| `tools/short_train.py` | Short scripted runs comparing baseline vs SSF loss curves on a tiny synthetic model (integration smoke test). |

## End-to-End Data Flow (Training)

1. YAML merges into `cfg` (`config` + `configs/...`).
2. `make_model` builds `build_transformer` or `build_transformer_local` with SSF flags from `cfg.PEFT.SSF`.
3. If `FREEZE_BACKBONE` is on, only parameters with `'ssf'` in the name remain trainable in `base` (and `b1`/`b2` for JPM); classifier and bottleneck remain trainable as usual.
4. `make_optimizer` assigns SSF-specific LR and zero weight decay.
5. Standard training scripts optimize ID + metric losses as in the rest of TransReID.

## File Index (Quick Reference)

| Path | Responsibility |
|------|------------------|
| `model/peft/ssf.py` | `SSF` module; merge/unmerge helpers. |
| `model/peft/__init__.py` | PEFT exports. |
| `model/backbones/vit_pytorch.py` | `Block` + `TransReID` SSF placements; factory functions. |
| `model/make_model.py` | Backbone construction; `_freeze_non_ssf`; JPM `deepcopy` + freeze order. |
| `config/defaults.py` | Default `PEFT.SSF` keys. |
| `solver/make_optimizer.py` | SSF LR multiplier; zero WD on SSF; AdamW param-group correctness. |
| `configs/Market/vit_transreid_ssf.yml` | Example SGD + SSF Market config. |
| `configs/Market/vit_transreid_stride_ssf_new.yml` | Example AdamW + paper-aligned notes. |
| `tests/test_ssf_integration.py` | Automated tests for SSF integration. |
| `tools/check_ssf.py` | Manual / CI SSF sanity check. |
| `tools/short_train.py` | Tiny-model training sanity check. |

## Summary

SSF-on-TransReID in this codebase means: **reusable `SSF` modules** (`model/peft/ssf.py`), **insertion at patch output, every selected transformer block (four sites each), and post-final-norm** (`model/backbones/vit_pytorch.py`), **config-driven enablement and block masking** (`config/defaults.py`, YAMLs), **backbone freezing by parameter name** (`model/make_model.py`), and **optimizer rules that give SSF a higher LR and no weight decay without breaking AdamW** (`solver/make_optimizer.py`), plus **tests and tools** to validate the stack.
