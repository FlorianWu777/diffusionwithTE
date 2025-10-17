# Sea Ice Forecasting Model Stack

This repository combines variational autoencoding, AFNO-based context encoding, and latent diffusion to forecast multi-month sea-ice and coupled climate fields. The stack is organized so that spatiotemporal data are normalized, encoded into a latent grid, iteratively denoised with classifier-free guidance, and decoded back to physical variables for evaluation and ensemble diagnostics.【F:train_gencfg.py†L24-L199】【F:models/diffusion/diffusion.py†L482-L599】

## Data representation and loading
- `ClimateForecastDataset` reads `.npy` files for multiple climate variables, normalizes them with precomputed statistics, and returns tensors shaped `[V, T, H, W]` for past inputs and `[V, T_out, H, W]` for prediction targets.【F:trainvqvae.py†L37-L120】【F:trainvqvae.py†L160-L196】  When `return_meta=True`, it also provides month indices per timestep so downstream modules can condition on the annual cycle.【F:trainvqvae.py†L160-L196】
- `SeaIceDataModule` splits CMIP model runs for transfer learning and ERA/OBS data for validation or fine-tuning, exposing Lightning dataloaders that honor distributed samplers and configurable future lead times.【F:train_gencfg.py†L24-L91】  Each sample therefore carries (i) past normalized fields (`x`), (ii) normalized futures (`y`), and (iii) metadata with `month_in`/`month_out` sequences.

## Temporal embedding
- `YearAdditiveEmbedding` learns a per-variable, per-month additive bias that is broadcast across the spatial grid. It uses actual month indices when provided and falls back to a relative position heuristic, while zero-valued indices disable the seasonal correction (used for classifier-free dropout).【F:annual_embedding.py†L4-L49】
- `AEWithTE` wraps any autoencoder to apply the additive bias before encoding and to remove it after decoding so seasonal statistics do not leak into unconditional branches. It optionally masks specific variables and delegates attribute lookups (e.g., `hidden_width`) to the wrapped model for compatibility with existing training code.【F:train_vae_embed.py†L242-L302】

## Autoencoder stage
- `ConvEncoder3D` downsamples the 3D climate cube (channels × time × lat × lon) through strided 3D convolutions and residual blocks, producing coarse latent means and log-variances at `H/16 × W/16` resolution. `ConvDecoder3D` mirrors the encoder with transposed convolutions and patch-wise attention to reconstruct the input volume from latents.【F:simpleautoencoder.py†L40-L143】
- `AutoencoderKL` turns the encoder/decoder pair into a variational module with scaling factor 0.18215, KL warm-up, and Lightning training hooks. It samples latents, decodes them back to the original resolution, and exposes `encode`/`decode` helpers used by the diffusion model.【F:simpleautoencoder.py†L228-L320】

## Context encoder (AFNO cascade)
- `AFNONowcastNetCascade` feeds the temporally embedded input context through an AFNO nowcasting backbone, building a multi-scale cascade of features keyed by spatial resolution. The wrapper applies the time embedding to the raw context tensor and seamlessly forwards attribute requests to the underlying AFNO model so diffusion modules can query channel dimensions.【F:models/genforecast/analysis.py†L10-L79】

### `analysis_net` 的作用

在训练与推理脚本中，`analysis_net` 实例化自 `AFNONowcastNetCascade`，承担以下职责：

1. **时间偏置预处理**：在把历史气候场送入 AFNO 主干网络前，先调用时间嵌入模块为输入加上按月份学习的季节性偏置，使时序上下文能够显式携带年循环信息。【F:models/genforecast/analysis.py†L34-L59】
2. **多尺度条件特征生成**：AFNO 主干会逐层下采样并通过 3D 残差块提取不同空间分辨率的特征图，形成一个以分辨率为键的字典，这些特征将作为扩散模型在各尺度的条件输入。【F:models/genforecast/analysis.py†L11-L33】
3. **属性转发与接口统一**：包装器会把外部对诸如 `cascade_dims`、`embed_dim_out` 等属性的访问转发给内部 AFNO 模型，保证后续构建扩散模型时能直接读取所需的通道配置，无需关心内部实现细节。【F:models/genforecast/analysis.py†L61-L79】【F:train_gencfg.py†L139-L196】

因此，`analysis_net` 是连接时序嵌入与扩散核心之间的关键桥梁，既对输入做季节性增强，又产出多尺度上下文，为噪声预测网络提供稳定的条件信号。【F:models/genforecast/analysis.py†L11-L79】【F:train_genforecast.py†L142-L189】

## Diffusion backbone
- `UNetModel` is a 3D UNet with residual blocks, attention at selected downsampling ratios, and timestep embeddings sized to the training horizon. It accepts latent channels from the autoencoder and produces noise/residual predictions matching the latent grid while consuming context channels supplied by the AFNO cascade.【F:models/genforecast/unet.py†L248-L360】
- `LatentDiffusion` orchestrates training and sampling: it freezes the autoencoder, registers the beta schedule, and implements classifier-free guidance by pairing conditional and zeroed contexts (including zeroed month indices).【F:models/diffusion/diffusion.py†L173-L420】  During each shared step it encodes future targets into latents, masks conditional inputs based on a dropout probability, and queries the context encoder before calling the UNet.【F:models/diffusion/diffusion.py†L482-L507】

## Training, monitoring, and inference outputs
- `setup_genforecast_training` packages the UNet, autoencoder, and context encoder into a Lightning `LatentDiffusion` module, configures early stopping and checkpoints, and chooses an accelerator strategy that respects the current world size. It also passes classifier-free dropout and guidance scale hyperparameters used at sampling time.【F:models/genforecast/training.py†L1-L74】
- `SpreadMonitor` draws multiple DDIM samples per validation batch, decodes them to `[B, V, T, H, W]` forecasts, and logs ensemble spread plus determinantal point process diversity metrics, illustrating the expected inference outputs of the pipeline.【F:models/diffusion/diffusion.py†L59-L130】  The autoencoder decode step reintroduces the seasonal bias if month metadata are supplied, yielding normalized climate fields ready for downstream denormalization or scoring.【F:models/diffusion/diffusion.py†L75-L109】

## Low-resolution direct diffusion pipeline
- `QuarterResolution` 下采样器在数据载入时把所有输入和目标的空间分辨率缩减为原来的 1/4，使得后续网络可以直接在像素空间建模而无需学习式编码器。【F:dataloader.py†L1-L32】
- `PastContextAdapter` 将 12 个月的历史多变量序列映射为与预测时间步对齐的上下文特征，为无自编码器的扩散模型提供条件输入。【F:models/direct/context.py†L1-L24】
- `Simple3DUNet` 以时间嵌入驱动的轻量 3D UNet，接受噪声目标与上下文拼接后的张量并输出去噪残差，适配低分辨率像素网格。【F:models/direct/unet.py†L1-L101】
- `DirectDiffusion` Lightning 模块把噪声调度、条件 dropout 与 DDIM 采样整合到像素级扩散流程中。【F:models/diffusion/direct_diffusion.py†L1-L141】
- `train_lowres_direct.py` 提供端到端训练脚本，构建数据模块、上下文编码器与扩散模型以预测未来 12 个月的海冰密集度。【F:train_lowres_direct.py†L1-L127】

## Component summary

| Component | Location | Inputs | Outputs | Purpose |
| --- | --- | --- | --- | --- |
| `YearAdditiveEmbedding` | `annual_embedding.py` | `[B, V, T, H, W]` tensor plus optional month indices | Bias-adjusted tensor with seasonal offsets | Injects learnable annual cycle without retraining core networks.【F:annual_embedding.py†L4-L49】 |
| `AEWithTE` | `train_vae_embed.py` | Climate tensor and month metadata | Autoencoder reconstructions & latents with bias removed | Ensures temporal embedding does not skew decoded outputs or unconditional branches.【F:train_vae_embed.py†L242-L302】 |
| `AutoencoderKL` | `simpleautoencoder.py` | Normalized climate cubes | Latent means/log-variances and reconstructions | Compresses inputs to latent grids consumed by diffusion.【F:simpleautoencoder.py†L84-L143】【F:simpleautoencoder.py†L228-L320】 |
| `AFNONowcastNetCascade` | `models/genforecast/analysis.py` | List containing context tensor, relative time, and months | Dict of multi-scale context feature maps | Supplies hierarchical conditioning signals to the UNet.【F:models/genforecast/analysis.py†L10-L67】 |
| `UNetModel` | `models/genforecast/unet.py` | Noisy latent grid, timestep embedding, AFNO context | Predicted noise or residual in latent space | Core denoiser for the diffusion process with 3D attention.【F:models/genforecast/unet.py†L248-L360】 |
| `LatentDiffusion` | `models/diffusion/diffusion.py` | Latent targets, conditioning context, beta schedule | Trained Lightning module with sampling & loss logic | Manages diffusion timesteps, CFG, losses, and ensemble diagnostics.【F:models/diffusion/diffusion.py†L173-L420】【F:models/diffusion/diffusion.py†L482-L599】 |

