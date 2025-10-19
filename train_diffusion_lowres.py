"""Training script for low-resolution diffusion forecasting without an autoencoder."""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from SICForecast import NoiseScheduleVP
from lowres_dataset import DownsampledClimateForecastDataset
from models.direct.unet import Simple3DUNet


def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule: NoiseScheduleVP) -> torch.Tensor:
    broadcast_shape = (t.shape[0],) + (1,) * (x_start.ndim - 1)
    alpha = schedule.marginal_mean_coeff(t).view(broadcast_shape)
    sigma = schedule.marginal_std(t).view(broadcast_shape)
    return alpha * x_start + sigma * noise


def histogram_kl_loss(pred: torch.Tensor, target: torch.Tensor, bins: int = 20, eps: float = 1e-8) -> torch.Tensor:
    batch = pred.shape[0]
    loss = pred.new_zeros(())
    for i in range(batch):
        pred_hist = torch.histc(pred[i], bins=bins, min=0.0, max=1.0)
        target_hist = torch.histc(target[i], bins=bins, min=0.0, max=1.0)
        pred_prob = pred_hist / (pred_hist.sum() + eps)
        target_prob = target_hist / (target_hist.sum() + eps)
        pred_prob = torch.clamp(pred_prob, min=eps)
        target_prob = torch.clamp(target_prob, min=eps)
        loss = loss + F.kl_div(pred_prob.log(), target_prob, reduction="sum")
    return loss / batch


def _parse_channel_multipliers(mult_string: str) -> tuple[int, ...]:
    values = [m.strip() for m in mult_string.split(",") if m.strip()]
    if not values:
        raise ValueError("channel-mults string must contain at least one integer")
    try:
        return tuple(int(v) for v in values)
    except ValueError as exc:
        raise ValueError(f"Invalid channel multiplier list: {mult_string}") from exc


def build_model(sample_x: torch.Tensor, sample_y: torch.Tensor, args: argparse.Namespace) -> Simple3DUNet:
    height, width = sample_y.shape[-2:]
    cond_channels = sample_x.shape[0] * sample_x.shape[1]

    if sample_y.ndim == 4:
        target_channels = sample_y.shape[0] * sample_y.shape[1]
    elif sample_y.ndim == 3:
        target_channels = sample_y.shape[0]
    else:
        raise ValueError(f"Unsupported target tensor rank: {sample_y.ndim}")

    in_channels = cond_channels + target_channels + 1  # +1 for the continuous time channel
    channel_multipliers = _parse_channel_multipliers(args.channel_mults)

    model = Simple3DUNet(
        in_channels=in_channels,
        out_channels=target_channels,
        base_channels=args.base_channels,
        channel_multipliers=channel_multipliers,
        norm_groups=args.norm_groups,
    )
    if args.use_data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    schedule: NoiseScheduleVP,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None,
    total_epochs: int,
    hist_loss_weight: float,
) -> float:
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_hist = 0.0
    num_samples = 0

    for x, y in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)
        num_samples += batch_size

        b, v, t_in, h, w = x.shape
        if y.ndim != 5:
            raise ValueError(f"Expected target tensor of rank 5, received shape {tuple(y.shape)}")

        target_vars = y.shape[1]
        t_out = y.shape[2]

        x_cond = x.permute(0, 2, 1, 3, 4).reshape(b, v * t_in, h, w)
        target = y.reshape(b, target_vars * t_out, h, w)

        noise = torch.randn_like(target)
        t_max = schedule.T * min(1.0, (epoch + 1) / total_epochs)
        t = torch.rand(batch_size, device=device) * t_max
        x_t = q_sample(target, t, noise, schedule)

        time_channel = t.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        model_input = torch.cat([x_cond, x_t, time_channel], dim=1)
        pred_noise = model(model_input)

        loss_denoise = F.mse_loss(pred_noise, noise)

        broadcast_shape = (batch_size, 1, 1, 1)
        alpha = schedule.marginal_mean_coeff(t).view(broadcast_shape)
        sigma = schedule.marginal_std(t).view(broadcast_shape)
        x0_pred = (x_t - sigma * pred_noise) / alpha
        loss_hist = histogram_kl_loss(x0_pred, target)

        loss = loss_denoise + hist_loss_weight * loss_hist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_mse += loss_denoise.item() * batch_size
        running_hist += loss_hist.item() * batch_size

    avg_loss = running_loss / num_samples
    avg_mse = running_mse / num_samples
    avg_hist = running_hist / num_samples

    if writer is not None:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/mse", avg_mse, epoch)
        writer.add_scalar("train/hist", avg_hist, epoch)

    return avg_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a low-resolution diffusion model for sea ice forecasting")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        required=True,
        help="Input variable directory names relative to each model root",
    )
    parser.add_argument("--target-var", type=str, default="siconca/abs", help="Target variable directory name")
    parser.add_argument("--mode", type=str, default="obs", choices=["transfer", "obs"], help="Dataset mode")
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="*",
        default=None,
        help="Specific model subdirectories when mode=transfer (e.g. EC-Earth3/r1i1p1f1)",
    )
    parser.add_argument("--input-seq-len", type=int, default=6, help="Number of historical months as input")
    parser.add_argument("--output-seq-len", type=int, default=12, help="Number of future months to predict")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hist-loss-weight", type=float, default=0.1)
    parser.add_argument("--scale-factor", type=float, default=0.25, help="Spatial downsampling factor relative to originals")
    parser.add_argument("--downsample-mode", type=str, default="bilinear")
    parser.add_argument("--base-channels", type=int, default=128, help="Base channel width for the UNet")
    parser.add_argument(
        "--channel-mults",
        type=str,
        default="1,2,4",
        help="Comma-separated channel multipliers for successive UNet stages",
    )
    parser.add_argument("--norm-groups", type=int, default=8, help="GroupNorm groups used in the UNet")
    parser.add_argument("--use-data-parallel", action="store_true")
    parser.add_argument("--noise-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None, help="Where to save the trained weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DownsampledClimateForecastDataset(
        root_dir=args.root_dir,
        variables=args.variables,
        target_var=[args.target_var],
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        mode=args.mode,
        model_names=args.model_names,
        scale_factor=args.scale_factor,
        downsample_mode=args.downsample_mode,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sample_x, sample_y = dataset[0]
    model = build_model(sample_x, sample_y, args)
    model = model.to(device)

    schedule = NoiseScheduleVP(schedule=args.noise_schedule)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    best_loss = float("inf")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(
            model,
            dataloader,
            optimizer,
            schedule,
            device,
            epoch,
            writer,
            args.epochs,
            args.hist_loss_weight,
        )
        if avg_loss < best_loss and args.save_path:
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, args.save_path)
            best_loss = avg_loss

    if writer is not None:
        writer.close()

    if args.save_path and best_loss < float("inf"):
        print(f"Training complete. Best model saved to {args.save_path}")


if __name__ == "__main__":
    main()
