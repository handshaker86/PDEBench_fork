"""Evaluation metrics for PDE benchmarks (RMSE, L2, spectral, vorticity, etc.)."""

from __future__ import annotations

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def metric_func(
    pred, target, if_mean=True, Lx=1.0, Ly=1.0, Lz=1.0, iLow=4, iHigh=12, initial_step=1
):
    """Compute metrics over full volume (batch + space + time). Returns RMSE, nRMSE, L2, max, spectral, vorticity, divergence."""
    device = pred.device
    target = target.to(device)
    epsilon = 1e-8

    # Align to (B, C, spatial..., T)
    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    elif len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    nc = idxs[1]

    diff = pred - target
    mse_global = torch.mean(diff**2)
    err_RMSE = torch.sqrt(mse_global)
    val_max = torch.max(target)
    val_min = torch.min(target)
    err_nRMSE = err_RMSE / (val_max - val_min + epsilon)

    sum_diff_sq = torch.sum(diff**2)
    sum_true_sq = torch.sum(target**2)
    err_rel_L2 = torch.sqrt(sum_diff_sq) / (torch.sqrt(sum_true_sq) + epsilon)

    err_Max = torch.max(torch.abs(diff))

    if len(idxs) == 4:
        fft_dims = [2]
    elif len(idxs) == 5:
        fft_dims = [2, 3]
    else:
        fft_dims = [2, 3, 4]
    pred_F = torch.fft.fftn(pred, dim=fft_dims)
    target_F = torch.fft.fftn(target, dim=fft_dims)
    psd_pred = torch.abs(pred_F)
    psd_target = torch.abs(target_F)
    log_diff = torch.log(psd_pred + epsilon) - torch.log(psd_target + epsilon)
    err_spec = torch.sqrt(torch.mean(log_diff**2))

    err_F = torch.tensor(0.0).to(device)
    err_vort = torch.tensor(0.0).to(device)
    err_div = torch.tensor(0.0).to(device)

    if nc >= 2 and len(idxs) >= 5:
        if len(idxs) == 5:
            dx, dy = Lx / idxs[2], Ly / idxs[3]
            u_p, v_p = pred[:, 0], pred[:, 1]
            u_t, v_t = target[:, 0], target[:, 1]
            dv_dx_p = torch.gradient(v_p, spacing=dx, dim=1)[0]
            du_dy_p = torch.gradient(u_p, spacing=dy, dim=2)[0]
            vor_pred = dv_dx_p - du_dy_p
            dv_dx_t = torch.gradient(v_t, spacing=dx, dim=1)[0]
            du_dy_t = torch.gradient(u_t, spacing=dy, dim=2)[0]
            vor_true = dv_dx_t - du_dy_t
            div_pred = (
                torch.gradient(u_p, spacing=dx, dim=1)[0]
                + torch.gradient(v_p, spacing=dy, dim=2)[0]
            )
            div_true = (
                torch.gradient(u_t, spacing=dx, dim=1)[0]
                + torch.gradient(v_t, spacing=dy, dim=2)[0]
            )

        elif len(idxs) == 6:
            dx, dy, dz = Lx / idxs[2], Ly / idxs[3], Lz / idxs[4]
            u_p, v_p, w_p = pred[:, 0], pred[:, 1], pred[:, 2]
            u_t, v_t, w_t = target[:, 0], target[:, 1], target[:, 2]
            div_pred = (
                torch.gradient(u_p, spacing=dx, dim=1)[0]
                + torch.gradient(v_p, spacing=dy, dim=2)[0]
                + torch.gradient(w_p, spacing=dz, dim=3)[0]
            )
            div_true = (
                torch.gradient(u_t, spacing=dx, dim=1)[0]
                + torch.gradient(v_t, spacing=dy, dim=2)[0]
                + torch.gradient(w_t, spacing=dz, dim=3)[0]
            )
            vor_pred = div_pred
            vor_true = div_true

        vor_sum_diff_sq = torch.sum((vor_true - vor_pred) ** 2)
        vor_sum_true_sq = torch.sum(vor_true**2)
        err_vort = torch.sqrt(vor_sum_diff_sq) / (torch.sqrt(vor_sum_true_sq) + epsilon)
        err_div = torch.sqrt(torch.mean((div_pred - div_true) ** 2))

    err_CSV = torch.tensor(0.0).to(device)
    err_BD = torch.tensor(0.0).to(device)

    return (
        err_RMSE,
        err_nRMSE,
        err_CSV,
        err_Max,
        err_BD,
        err_F,
        err_rel_L2,
        err_vort,
        err_div,
        err_spec,
    )


def metrics(
    val_loader,
    model,
    Lx,
    Ly,
    Lz,
    plot,
    channel_plot,
    model_name,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max,
    mode="FNO",
    initial_step=None,
    prediction_step=None,
    result_save_path=None,
    warmup_runs: int = 5,
    average_runs: int = 10,
    rollout_plot: bool = False,
):
    num_runs = max(1, average_runs)
    all_total_times = []
    rollout_metric_time = None
    rollout_metric_steps = None

    if mode == "Unet":
        with torch.no_grad():
            print("Warming up GPU...")
            for _ in range(warmup_runs):
                xx, yy = next(iter(val_loader))
                xx = xx.to(device)  # noqa: PLW2901
                yy = yy.to(device)  # noqa: PLW2901
                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)
                inp = xx.reshape(inp_shape)
                shape = [0, -1]
                shape.extend(list(range(1, len(inp.shape) - 1)))
                inp = inp.permute(shape)
                im = model(inp)
            torch.cuda.synchronize()
            print(f"GPU is ready.")

            for run_idx in range(num_runs):
                for itot, (xx, yy) in enumerate(val_loader):
                    xx = xx.to(device)  # noqa: PLW2901
                    yy = yy.to(device)  # noqa: PLW2901

                    pred = yy[..., :initial_step, :]
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    T = yy.shape[-2]
                    if rollout_metric_time is None or itot == 0:
                        rollout_metric_steps = torch.arange(
                            initial_step, T, prediction_step, device=device
                        )
                        n_rollouts = int(rollout_metric_steps.numel())
                        rollout_metric_time = {
                            "RMSE": torch.zeros(n_rollouts, device=device),
                            "nRMSE": torch.zeros(n_rollouts, device=device),
                            "CSV": torch.zeros(n_rollouts, device=device),
                            "Max": torch.zeros(n_rollouts, device=device),
                            "BD": torch.zeros(n_rollouts, device=device),
                            "F": torch.zeros(n_rollouts, device=device),
                            "rel_L2": torch.zeros(n_rollouts, device=device),
                            "vort": torch.zeros(n_rollouts, device=device),
                            "div": torch.zeros(n_rollouts, device=device),
                            "spec": torch.zeros(n_rollouts, device=device),
                        }

                    rollout_idx = 0
                    for t in range(initial_step, T, prediction_step):
                        inp = xx.reshape(inp_shape)
                        shape = [0, -1]
                        shape.extend(list(range(1, len(inp.shape) - 1)))
                        inp = inp.permute(shape)
                        shape = [0]
                        shape.extend(list(range(2, len(inp.shape))))
                        shape.append(1)
                        im = model(inp)
                        pred = torch.cat((pred, im), -2)
                        xx = torch.cat(
                            (xx[..., prediction_step:, :], im), dim=-2
                        )  # noqa: PLW2901

                        _pred_one = im[..., 0:1, :]
                        _yy_one = yy[..., t : (t + 1), :]
                        (
                            _r_rmse,
                            _r_nrmse,
                            _r_csv,
                            _r_max,
                            _r_bd,
                            _r_f,
                            _r_rel_l2,
                            _r_vort,
                            _r_div,
                            _r_spec,
                        ) = metric_func(
                            _pred_one,
                            _yy_one,
                            if_mean=True,
                            Lx=Lx,
                            Ly=Ly,
                            Lz=Lz,
                            initial_step=initial_step,
                        )
                        if rollout_idx < rollout_metric_steps.numel():
                            rollout_metric_time["RMSE"][rollout_idx] += _r_rmse
                            rollout_metric_time["nRMSE"][rollout_idx] += _r_nrmse
                            rollout_metric_time["CSV"][rollout_idx] += _r_csv
                            rollout_metric_time["Max"][rollout_idx] += _r_max
                            rollout_metric_time["BD"][rollout_idx] += _r_bd
                            rollout_metric_time["F"][rollout_idx] += _r_f
                            rollout_metric_time["rel_L2"][rollout_idx] += _r_rel_l2
                            rollout_metric_time["vort"][rollout_idx] += _r_vort
                            rollout_metric_time["div"][rollout_idx] += _r_div
                            rollout_metric_time["spec"][rollout_idx] += _r_spec
                        rollout_idx += 1

                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    num_frames = yy.shape[-2] - initial_step
                    prediction_time = end_time - start_time

                    pred_len = min(pred.shape[-2], T)
                    _pred_truncated = pred[..., initial_step:pred_len, :]
                    _yy_truncated = yy[..., initial_step:pred_len, :]

                    (
                        _err_RMSE,
                        _err_nRMSE,
                        _err_CSV,
                        _err_Max,
                        _err_BD,
                        _err_F,
                        _err_rel_L2,
                        _err_vort,
                        _err_div,
                        _err_spec,
                    ) = metric_func(
                        _pred_truncated,
                        _yy_truncated,
                        if_mean=True,
                        Lx=Lx,
                        Ly=Ly,
                        Lz=Lz,
                        initial_step=initial_step,
                    )

                    if itot == 0:
                        (
                            err_RMSE,
                            err_nRMSE,
                            err_CSV,
                            err_Max,
                            err_BD,
                            err_F,
                            err_rel_L2,
                            err_vort,
                            err_div,
                            err_spec,
                        ) = (
                            _err_RMSE,
                            _err_nRMSE,
                            _err_CSV,
                            _err_Max,
                            _err_BD,
                            _err_F,
                            _err_rel_L2,
                            _err_vort,
                            _err_div,
                            _err_spec,
                        )
                        total_time = prediction_time
                        pred_plot = pred[:1]
                        target_plot = yy[:1]
                        val_l2_time = torch.zeros(yy.shape[-2]).to(device)
                    else:
                        err_RMSE += _err_RMSE
                        err_nRMSE += _err_nRMSE
                        err_CSV += _err_CSV
                        err_Max += _err_Max
                        err_BD += _err_BD
                        err_F += _err_F
                        err_rel_L2 += _err_rel_L2
                        err_vort += _err_vort
                        err_div += _err_div
                        err_spec += _err_spec

                        total_time += prediction_time

                        mean_dim = list(range(len(yy.shape) - 2))
                        mean_dim.append(-1)
                        mean_dim = tuple(mean_dim)

                        _val_l2_per_step = torch.sqrt(
                            torch.mean(
                                (_pred_truncated - _yy_truncated) ** 2, dim=mean_dim
                            )
                        )
                        val_l2_time[: (pred_len - initial_step)] += _val_l2_per_step

                all_total_times.append(total_time)

            total_time = sum(all_total_times) / len(all_total_times)

    elif mode == "FNO":
        with torch.no_grad():
            print("Warming up GPU...")
            for _ in range(warmup_runs):
                xx, yy, grid = next(iter(val_loader))
                xx = xx.to(device)  # noqa: PLW2901
                yy = yy.to(device)  # noqa: PLW2901
                grid = grid.to(device)  # noqa: PLW2901
                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)
                inp = xx.reshape(inp_shape)
                im = model(inp, grid)
            torch.cuda.synchronize()
            print(f"GPU is ready.")

            for run_idx in range(num_runs):
                for itot, (xx, yy, grid) in enumerate(val_loader):
                    xx = xx.to(device)  # noqa: PLW2901
                    yy = yy.to(device)  # noqa: PLW2901
                    grid = grid.to(device)  # noqa: PLW2901

                    pred = yy[..., :initial_step, :]
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    T = yy.shape[-2]
                    if rollout_metric_time is None or itot == 0:
                        rollout_metric_steps = torch.arange(
                            initial_step, T, prediction_step, device=device
                        )
                        n_rollouts = int(rollout_metric_steps.numel())
                        rollout_metric_time = {
                            "RMSE": torch.zeros(n_rollouts, device=device),
                            "nRMSE": torch.zeros(n_rollouts, device=device),
                            "CSV": torch.zeros(n_rollouts, device=device),
                            "Max": torch.zeros(n_rollouts, device=device),
                            "BD": torch.zeros(n_rollouts, device=device),
                            "F": torch.zeros(n_rollouts, device=device),
                            "rel_L2": torch.zeros(n_rollouts, device=device),
                            "vort": torch.zeros(n_rollouts, device=device),
                            "div": torch.zeros(n_rollouts, device=device),
                            "spec": torch.zeros(n_rollouts, device=device),
                        }

                    rollout_idx = 0
                    for t in range(initial_step, T, prediction_step):
                        inp = xx.reshape(inp_shape)
                        im = model(inp, grid)
                        pred = torch.cat((pred, im), -2)
                        xx = torch.cat(
                            (xx[..., prediction_step:, :], im), dim=-2
                        )  # noqa: PLW2901

                        _pred_one = im[..., 0:1, :]
                        _yy_one = yy[..., t : (t + 1), :]
                        (
                            _r_rmse,
                            _r_nrmse,
                            _r_csv,
                            _r_max,
                            _r_bd,
                            _r_f,
                            _r_rel_l2,
                            _r_vort,
                            _r_div,
                            _r_spec,
                        ) = metric_func(
                            _pred_one,
                            _yy_one,
                            if_mean=True,
                            Lx=Lx,
                            Ly=Ly,
                            Lz=Lz,
                            initial_step=initial_step,
                        )
                        if rollout_idx < rollout_metric_steps.numel():
                            rollout_metric_time["RMSE"][rollout_idx] += _r_rmse
                            rollout_metric_time["nRMSE"][rollout_idx] += _r_nrmse
                            rollout_metric_time["CSV"][rollout_idx] += _r_csv
                            rollout_metric_time["Max"][rollout_idx] += _r_max
                            rollout_metric_time["BD"][rollout_idx] += _r_bd
                            rollout_metric_time["F"][rollout_idx] += _r_f
                            rollout_metric_time["rel_L2"][rollout_idx] += _r_rel_l2
                            rollout_metric_time["vort"][rollout_idx] += _r_vort
                            rollout_metric_time["div"][rollout_idx] += _r_div
                            rollout_metric_time["spec"][rollout_idx] += _r_spec
                        rollout_idx += 1

                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    num_frames = yy.shape[-2] - initial_step
                    prediction_time = end_time - start_time

                    pred_len = min(pred.shape[-2], T)
                    _pred_truncated = pred[..., initial_step:pred_len, :]
                    _yy_truncated = yy[..., initial_step:pred_len, :]
                    (
                        _err_RMSE,
                        _err_nRMSE,
                        _err_CSV,
                        _err_Max,
                        _err_BD,
                        _err_F,
                        _err_rel_L2,
                        _err_vort,
                        _err_div,
                        _err_spec,
                    ) = metric_func(
                        _pred_truncated,
                        _yy_truncated,
                        if_mean=True,
                        Lx=Lx,
                        Ly=Ly,
                        Lz=Lz,
                        initial_step=initial_step,
                    )

                    if itot == 0:
                        (
                            err_RMSE,
                            err_nRMSE,
                            err_CSV,
                            err_Max,
                            err_BD,
                            err_F,
                            err_rel_L2,
                            err_vort,
                            err_div,
                            err_spec,
                        ) = (
                            _err_RMSE,
                            _err_nRMSE,
                            _err_CSV,
                            _err_Max,
                            _err_BD,
                            _err_F,
                            _err_rel_L2,
                            _err_vort,
                            _err_div,
                            _err_spec,
                        )
                        pred_plot = pred[:1]
                        target_plot = yy[:1]
                        val_l2_time = torch.zeros(yy.shape[-2]).to(device)
                        total_time = prediction_time
                    else:
                        err_RMSE += _err_RMSE
                        err_nRMSE += _err_nRMSE
                        err_CSV += _err_CSV
                        err_Max += _err_Max
                        err_BD += _err_BD
                        err_F += _err_F
                        err_rel_L2 += _err_rel_L2
                        err_vort += _err_vort
                        err_div += _err_div
                        err_spec += _err_spec

                        total_time += prediction_time

                        mean_dim = list(range(len(yy.shape) - 2))
                        mean_dim.append(-1)
                        mean_dim = tuple(mean_dim)

                        _val_l2_per_step = torch.sqrt(
                            torch.mean(
                                (_pred_truncated - _yy_truncated) ** 2, dim=mean_dim
                            )
                        )
                        val_l2_time[: (pred_len - initial_step)] += _val_l2_per_step

                all_total_times.append(total_time)

            total_time = sum(all_total_times) / len(all_total_times)

    elif mode == "PINN":
        raise NotImplementedError

    err_RMSE = np.array(err_RMSE.data.cpu() / (itot + 1))
    err_nRMSE = np.array(err_nRMSE.data.cpu() / (itot + 1))
    err_CSV = np.array(err_CSV.data.cpu() / (itot + 1))
    err_Max = np.array(err_Max.data.cpu() / (itot + 1))
    err_BD = np.array(err_BD.data.cpu() / (itot + 1))
    err_F = np.array(err_F.data.cpu() / (itot + 1))
    err_rel_L2 = np.array(err_rel_L2.data.cpu() / (itot + 1))
    err_vort = np.array(err_vort.data.cpu() / (itot + 1))
    err_div = np.array(err_div.data.cpu() / (itot + 1))
    err_spec = np.array(err_spec.data.cpu() / (itot + 1))

    batch_prediction_time = total_time / (itot + 1)
    frame_prediction_time = batch_prediction_time / num_frames

    logger.info(f"RMSE: {err_RMSE:.5f}")
    logger.info(f"normalized RMSE: {err_nRMSE:.5f}")
    logger.info(f"RMSE of conserved variables: {err_CSV:.5f}")
    logger.info(f"Maximum value of rms error: {err_Max:.5f}")
    logger.info(f"RMSE at boundaries: {err_BD:.5f}")
    logger.info(f"RMSE in Fourier space: {err_F}")
    logger.info(f"Average Prediction time (per batch): {batch_prediction_time:.5f}")
    logger.info(f"Relative L2 error: {err_rel_L2:.5f}")
    logger.info(f"Vorticity error: {err_vort:.5f}")
    logger.info(f"Divergence error: {err_div:.5f}")
    logger.info(f"Spectral error: {err_spec:.5f}")

    val_l2_time = val_l2_time / (itot + 1)

    os.makedirs(result_save_path, exist_ok=True)
    with open(result_save_path + "loss.txt", "w") as f:
        f.write(
            f"RMSE: {err_RMSE:.5f}\n"
            f"normalized RMSE: {err_nRMSE:.5f}\n"
            f"RMSE of conserved variables: {err_CSV:.5f}\n"
            f"Maximum value of rms error: {err_Max:.5f}\n"
            f"RMSE at boundaries: {err_BD:.5f}\n"
            f"RMSE in Fourier space: {err_F}\n"
            f"Relative L2 error: {err_rel_L2:.5f}\n"
            f"Vorticity error: {err_vort:.5f}\n"
            f"Divergence error: {err_div:.5f}\n"
            f"Spectral error: {err_spec:.5f}\n"
        )

    with open(result_save_path + "predict_time.txt", "w") as f:
        f.write(f"total prediction time: {total_time:.5f}\n")
        f.write(f"prediction time for each batch: {batch_prediction_time:.5f}\n")
        f.write(f"prediction time for each frame: {frame_prediction_time:.5f}\n")

    if rollout_metric_time is not None and rollout_metric_steps is not None:
        rollout_metric_time = {
            k: (v / (itot + 1)).detach().cpu().numpy()
            for k, v in rollout_metric_time.items()
        }
        rollout_steps_cpu = rollout_metric_steps.detach().cpu().numpy()
        n_rollouts_np = int(len(rollout_steps_cpu))
        rollout_step_rel = np.arange(n_rollouts_np, dtype=np.int32) * prediction_step
        rollout_total_predicted = rollout_step_rel.copy()

        rollout_csv_path = os.path.join(result_save_path, "rollout_result.csv")
        with open(rollout_csv_path, "w") as f:
            f.write(
                "idx,rollout_step,total_rollout_step,"
                "RMSE,nRMSE,CSV,Max,BD,F,rel_L2,vort,div,spec\n"
            )
            n = int(len(rollout_steps_cpu))
            for i in range(n):
                f.write(
                    f"{i},{int(rollout_steps_cpu[i])},{int(rollout_total_predicted[i])},"
                    f"{rollout_metric_time['RMSE'][i]:.6e},"
                    f"{rollout_metric_time['nRMSE'][i]:.6e},"
                    f"{rollout_metric_time['CSV'][i]:.6e},"
                    f"{rollout_metric_time['Max'][i]:.6e},"
                    f"{rollout_metric_time['BD'][i]:.6e},"
                    f"{rollout_metric_time['F'][i]:.6e},"
                    f"{rollout_metric_time['rel_L2'][i]:.6e},"
                    f"{rollout_metric_time['vort'][i]:.6e},"
                    f"{rollout_metric_time['div'][i]:.6e},"
                    f"{rollout_metric_time['spec'][i]:.6e}\n"
                )

        if rollout_plot:
            plt.ioff()
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            axes = axes.flatten()
            x = rollout_total_predicted
            axes[0].plot(x, rollout_metric_time["RMSE"], label="RMSE")
            axes[0].plot(x, rollout_metric_time["nRMSE"], label="nRMSE")
            axes[0].set_title("RMSE / nRMSE (1st frame per rollout)")
            axes[0].legend(fontsize=10)

            axes[1].plot(x, rollout_metric_time["rel_L2"], label="rel_L2")
            axes[1].plot(x, rollout_metric_time["Max"], label="Max")
            axes[1].set_title("rel_L2 / Max (1st frame per rollout)")
            axes[1].legend(fontsize=10)

            axes[2].plot(x, rollout_metric_time["vort"], label="vort")
            axes[2].plot(x, rollout_metric_time["div"], label="div")
            axes[2].set_title("vort / div (1st frame per rollout)")
            axes[2].legend(fontsize=10)

            axes[3].plot(x, rollout_metric_time["spec"], label="spec")
            axes[3].set_title("spec (1st frame per rollout)")
            axes[3].legend(fontsize=10)
            for ax in axes:
                ax.set_xlabel("total rollout step")
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(result_save_path, "rollout_metrics_first_step.pdf")
            )

    if plot:
        dim = len(yy.shape) - 3
        plt.ioff()
        if dim == 1:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(
                pred_plot[..., channel_plot].squeeze().detach().cpu(),
                extent=[t_min, t_max, x_min, x_max],
                origin="lower",
                aspect="auto",
            )
            h.set_clim(
                target_plot[..., channel_plot].min(),
                target_plot[..., channel_plot].max(),
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Prediction", fontsize=30)
            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.set_ylabel("$x$", fontsize=30)
            ax.set_xlabel("$t$", fontsize=30)
            plt.tight_layout()
            filename = model_name + "_pred.pdf"
            plt.savefig(filename)

            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(
                target_plot[..., channel_plot].squeeze().detach().cpu(),
                extent=[t_min, t_max, x_min, x_max],
                origin="lower",
                aspect="auto",
            )
            h.set_clim(
                target_plot[..., channel_plot].min(),
                target_plot[..., channel_plot].max(),
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Data", fontsize=30)
            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.set_ylabel("$x$", fontsize=30)
            ax.set_xlabel("$t$", fontsize=30)
            plt.tight_layout()
            filename = model_name + "_data.pdf"
            plt.savefig(filename)

        elif dim == 2:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(
                pred_plot[..., -1, channel_plot].squeeze().t().detach().cpu(),
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                aspect="auto",
            )
            h.set_clim(
                target_plot[..., -1, channel_plot].min(),
                target_plot[..., -1, channel_plot].max(),
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Prediction", fontsize=30)
            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.set_ylabel("$y$", fontsize=30)
            ax.set_xlabel("$x$", fontsize=30)
            plt.tight_layout()
            filename = model_name + "_pred.pdf"
            plt.savefig(filename)

            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(
                target_plot[..., -1, channel_plot].squeeze().t().detach().cpu(),
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                aspect="auto",
            )
            h.set_clim(
                target_plot[..., -1, channel_plot].min(),
                target_plot[..., -1, channel_plot].max(),
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Data", fontsize=30)
            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.set_ylabel("$y$", fontsize=30)
            ax.set_xlabel("$x$", fontsize=30)
            plt.tight_layout()
            filename = model_name + "_data.pdf"
            plt.savefig(filename)

        filename = model_name + "mse_time.npz"
        np.savez(
            filename,
            t=torch.arange(initial_step, yy.shape[-2]).cpu(),
            mse=val_l2_time[initial_step:].detach().cpu(),
        )

    return (
        err_RMSE,
        err_nRMSE,
        err_CSV,
        err_Max,
        err_BD,
        err_F,
        err_rel_L2,
        err_vort,
        err_div,
        err_spec,
    )


def save_prediction_results(
    val_loader,
    model,
    mode="FNO",
    initial_step=None,
    prediction_step=None,
    model_name=None,
    dataset_name=None,
):
    """Run prediction and save as .npz with keys frame_i -> (u, v) arrays."""
    if model_name is None:
        model_name = mode
    if dataset_name is None:
        dataset_name = "unknown"

    final_pred = None
    all_preds = []

    if mode == "Unet":
        with torch.no_grad():
            for itot, (xx, yy) in enumerate(val_loader):
                xx = xx.to(device)
                yy = yy.to(device)

                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                T = yy.shape[-2]

                for t in range(initial_step, T, prediction_step):
                    inp = xx.reshape(inp_shape)
                    shape = [0, -1]
                    shape.extend(list(range(1, len(inp.shape) - 1)))
                    inp = inp.permute(shape)
                    shape = [0]
                    shape.extend(list(range(2, len(inp.shape))))
                    shape.append(1)
                    im = model(inp)
                    pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., prediction_step:, :], im), dim=-2)

                pred_length = min(pred.shape[-2], T)
                all_preds.append(
                    pred[..., initial_step:pred_length, :].detach().cpu().numpy()
                )

    elif mode == "FNO":
        with torch.no_grad():
            for itot, (xx, yy, grid) in enumerate(val_loader):
                xx = xx.to(device)
                yy = yy.to(device)
                grid = grid.to(device)

                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                T = yy.shape[-2]

                for t in range(initial_step, T, prediction_step):
                    inp = xx.reshape(inp_shape)
                    im = model(inp, grid)
                    pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., prediction_step:, :], im), dim=-2)
                pred_length = min(pred.shape[-2], T)
                all_preds.append(
                    pred[..., initial_step:pred_length, :].detach().cpu().numpy()
                )

    elif mode == "PINN":
        raise NotImplementedError(
            "PINN mode not supported for saving prediction results"
        )

    if all_preds:
        final_pred = np.concatenate(all_preds, axis=0)

    if final_pred is not None:
        pred_shape = final_pred.shape
        ndim = len(pred_shape)

        if pred_shape[-1] < 2:
            logger.warning(
                f"Prediction has only {pred_shape[-1]} channel(s), need at least 2 channels for (u, v)"
            )
            return None

        if ndim == 5:
            u_data = final_pred[..., 0]
            v_data = final_pred[..., 1]
            u_data = np.transpose(u_data, (0, 3, 1, 2))
            v_data = np.transpose(v_data, (0, 3, 1, 2))
            # Merge batch and frame: (B*T, X, Y)
            num_batches, num_frames_per_sample = u_data.shape[0], u_data.shape[1]
            u_data = u_data.reshape(
                num_batches * num_frames_per_sample, *u_data.shape[2:]
            )
            v_data = v_data.reshape(
                num_batches * num_frames_per_sample, *v_data.shape[2:]
            )
            num_frames = u_data.shape[0]
        elif ndim == 4:
            u_data = final_pred[..., 0]
            v_data = final_pred[..., 1]
            u_data = np.transpose(u_data, (0, 2, 1))
            v_data = np.transpose(v_data, (0, 2, 1))
            num_batches, num_frames_per_sample = u_data.shape[0], u_data.shape[1]
            u_data = u_data.reshape(
                num_batches * num_frames_per_sample, u_data.shape[2], 1
            )
            v_data = v_data.reshape(
                num_batches * num_frames_per_sample, v_data.shape[2], 1
            )
            num_frames = u_data.shape[0]
        elif ndim == 6:
            z_mid = pred_shape[3] // 2
            u_data = final_pred[:, :, :, z_mid, :, 0]
            v_data = final_pred[:, :, :, z_mid, :, 1]
            u_data = np.transpose(u_data, (0, 3, 1, 2))
            v_data = np.transpose(v_data, (0, 3, 1, 2))
            num_batches, num_frames_per_sample = u_data.shape[0], u_data.shape[1]
            u_data = u_data.reshape(
                num_batches * num_frames_per_sample, *u_data.shape[2:]
            )
            v_data = v_data.reshape(
                num_batches * num_frames_per_sample, *v_data.shape[2:]
            )
            num_frames = u_data.shape[0]
        else:
            logger.warning(f"Unexpected prediction shape: {pred_shape}, skipping save")
            return None

        save_dict = {}
        for i in range(num_frames):
            save_dict[f"frame_{i}"] = (u_data[i], v_data[i])

        save_dir = os.path.join("./prediction", model_name, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "prediction_result.npz")
        np.savez(save_path, **save_dict)
        logger.info(f"Prediction results saved to {save_path} with {num_frames} frames")
        logger.info(f"Each frame has shape: u={u_data[0].shape}, v={v_data[0].shape}")
        return save_path

    return None


def predict_time_benchmark(
    val_loader,
    model,
    mode="FNO",
    initial_step=None,
    prediction_step=None,
    target_frames=100,
    warmup_runs: int = 5,
    average_runs: int = 10,
    output_file="./results/benchmark/predict_time.txt",
    data_name=None,
):
    """Benchmark prediction time for target_frames; returns avg time per frame (seconds)."""
    num_runs = max(1, average_runs)
    all_times = []

    if mode == "Unet":
        with torch.no_grad():
            print("Warming up GPU...")
            for _ in range(warmup_runs):
                xx, yy = next(iter(val_loader))
                xx = xx.to(device)
                yy = yy.to(device)
                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)
                inp = xx.reshape(inp_shape)
                shape = [0, -1]
                shape.extend(list(range(1, len(inp.shape) - 1)))
                inp = inp.permute(shape)
                im = model(inp)
            torch.cuda.synchronize()
            print("GPU is ready.")

            for run_idx in range(num_runs):
                frames_predicted = 0
                total_time = 0.0

                for itot, (xx, yy) in enumerate(val_loader):
                    if frames_predicted >= target_frames:
                        break

                    xx = xx.to(device)
                    yy = yy.to(device)

                    pred = yy[..., :initial_step, :]
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    T = yy.shape[-2]

                    torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    for t in range(initial_step, T, prediction_step):
                        if frames_predicted >= target_frames:
                            break

                        inp = xx.reshape(inp_shape)
                        shape = [0, -1]
                        shape.extend(list(range(1, len(inp.shape) - 1)))
                        inp = inp.permute(shape)
                        shape = [0]
                        shape.extend(list(range(2, len(inp.shape))))
                        shape.append(1)
                        im = model(inp)
                        pred = torch.cat((pred, im), -2)
                        xx = torch.cat((xx[..., prediction_step:, :], im), dim=-2)
                        frames_predicted += prediction_step

                        if frames_predicted >= target_frames:
                            break

                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    prediction_time = end_time - start_time
                    total_time += prediction_time

                    if frames_predicted >= target_frames:
                        break

                all_times.append(total_time)

    elif mode == "FNO":
        with torch.no_grad():
            print("Warming up GPU...")
            for _ in range(warmup_runs):
                xx, yy, grid = next(iter(val_loader))
                xx = xx.to(device)
                yy = yy.to(device)
                grid = grid.to(device)
                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)
                inp = xx.reshape(inp_shape)
                im = model(inp, grid)
            torch.cuda.synchronize()
            print("GPU is ready.")

            for run_idx in range(num_runs):
                frames_predicted = 0
                total_time = 0.0

                for itot, (xx, yy, grid) in enumerate(val_loader):
                    if frames_predicted >= target_frames:
                        break

                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)

                    pred = yy[..., :initial_step, :]
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    T = yy.shape[-2]

                    torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    for t in range(initial_step, T, prediction_step):
                        if frames_predicted >= target_frames:
                            break

                        inp = xx.reshape(inp_shape)
                        im = model(inp, grid)
                        pred = torch.cat((pred, im), -2)
                        xx = torch.cat((xx[..., prediction_step:, :], im), dim=-2)
                        frames_predicted += prediction_step

                        if frames_predicted >= target_frames:
                            break

                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    prediction_time = end_time - start_time
                    total_time += prediction_time

                    if frames_predicted >= target_frames:
                        break

                all_times.append(total_time)

    elif mode == "PINN":
        raise NotImplementedError

    avg_time = sum(all_times) / len(all_times)
    avg_time_per_frame = avg_time / target_frames

    output_dir = os.path.dirname(output_file)
    output_filename = os.path.basename(output_file)
    if output_dir:
        if data_name:
            output_file = os.path.join(output_dir, mode, data_name, output_filename)
        else:
            output_file = os.path.join(output_dir, mode, output_filename)
    else:
        if data_name:
            output_file = os.path.join(mode, data_name, output_filename)
        else:
            output_file = os.path.join(mode, output_filename)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(
            f"Average prediction time for {target_frames} frames: {avg_time:.5f}s\n"
        )
        f.write(f"Average prediction time per frame: {avg_time_per_frame:.5f}s\n")

    return avg_time_per_frame


class LpLoss:
    """Lp-norm relative loss."""

    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        assert p > 0
        self.p = p
        self.reduction = reduction

    def __call__(self, x, y, eps=1e-20):
        num_examples = x.size()[0]
        _diff = x.view(num_examples, -1) - y.view(num_examples, -1)
        _diff = torch.norm(_diff, self.p, 1)
        _norm = eps + torch.norm(y.view(num_examples, -1), self.p, 1)
        if self.reduction in ["mean"]:
            return torch.mean(_diff / _norm)
        if self.reduction in ["sum"]:
            return torch.sum(_diff / _norm)
        return _diff / _norm


class FftLpLoss:
    """Lp loss in Fourier space."""

    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        assert p > 0
        self.p = p
        self.reduction = reduction

    def __call__(self, x, y, flow=None, fhigh=None, eps=1e-20):
        num_examples = x.size()[0]
        others_dims = x.shape[1:]
        dims = list(range(1, len(x.shape)))
        xf = torch.fft.fftn(x, dim=dims)
        yf = torch.fft.fftn(y, dim=dims)
        if flow is None:
            flow = 0
        if fhigh is None:
            fhigh = np.max(xf.shape[1:])

        if len(others_dims) == 1:
            xf = xf[:, flow:fhigh]
            yf = yf[:, flow:fhigh]
        if len(others_dims) == 2:
            xf = xf[:, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh]
        if len(others_dims) == 3:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh]
        if len(others_dims) == 4:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]

        _diff = xf - yf.reshape(xf.shape)
        _diff = torch.norm(_diff.reshape(num_examples, -1), self.p, 1)
        _norm = eps + torch.norm(yf.reshape(num_examples, -1), self.p, 1)

        if self.reduction in ["mean"]:
            return torch.mean(_diff / _norm)
        if self.reduction in ["sum"]:
            return torch.sum(_diff / _norm)
        return _diff / _norm


class FftMseLoss:
    """MSE loss in Fourier space."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def __call__(self, x, y, flow=None, fhigh=None):
        num_examples = x.size()[0]
        others_dims = x.shape[1:-2]
        for d in others_dims:
            assert d > 1, "dimension must be > 1"
        dims = list(range(1, len(x.shape) - 1))
        xf = torch.fft.fftn(x, dim=dims)
        yf = torch.fft.fftn(y, dim=dims)
        if flow is None:
            flow = 0
        if fhigh is None:
            fhigh = np.max(xf.shape[1:])

        if len(others_dims) == 1:
            xf = xf[:, flow:fhigh]
            yf = yf[:, flow:fhigh]
        if len(others_dims) == 2:
            xf = xf[:, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh]
        if len(others_dims) == 3:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh]
        if len(others_dims) == 4:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]
        _diff = xf - yf
        _diff = _diff.reshape(num_examples, -1).abs() ** 2
        if self.reduction in ["mean"]:
            return torch.mean(_diff).abs()
        if self.reduction in ["sum"]:
            return torch.sum(_diff).abs()
        return _diff.abs()


def inverse_metrics(u0, x, pred_u0, y):
    """Metrics in spatial and Fourier space: u0 vs x (IC), pred_u0 vs y (prediction)."""

    mseloss_fn = nn.MSELoss(reduction="mean")
    l2loss_fn = LpLoss(p=2, reduction="mean")
    l3loss_fn = LpLoss(p=3, reduction="mean")

    fftmseloss_fn = FftMseLoss(reduction="mean")
    fftl2loss_fn = FftLpLoss(p=2, reduction="mean")
    fftl3loss_fn = FftLpLoss(p=3, reduction="mean")

    mseloss_u0 = mseloss_fn(u0.view(1, -1), x.view(1, -1)).item()
    l2loss_u0 = l2loss_fn(u0.view(1, -1), x.view(1, -1)).item()
    l3loss_u0 = l3loss_fn(u0.view(1, -1), x.view(1, -1)).item()

    fmid = u0.shape[1] // 4

    fftmseloss_u0 = fftmseloss_fn(u0, x).item()
    fftmseloss_low_u0 = fftmseloss_fn(u0, x, 0, fmid).item()
    fftmseloss_mid_u0 = fftmseloss_fn(u0, x, fmid, 2 * fmid).item()
    fftmseloss_hi_u0 = fftmseloss_fn(u0, x, 2 * fmid).item()

    fftl2loss_u0 = fftl2loss_fn(u0, x).item()
    fftl2loss_low_u0 = fftl2loss_fn(u0, x, 0, fmid).item()
    fftl2loss_mid_u0 = fftl2loss_fn(u0, x, fmid, 2 * fmid).item()
    fftl2loss_hi_u0 = fftl2loss_fn(u0, x, 2 * fmid).item()

    fftl3loss_u0 = fftl3loss_fn(u0, x).item()
    fftl3loss_low_u0 = fftl3loss_fn(u0, x, 0, fmid).item()
    fftl3loss_mid_u0 = fftl3loss_fn(u0, x, fmid, 2 * fmid).item()
    fftl3loss_hi_u0 = fftl3loss_fn(u0, x, 2 * fmid).item()

    mseloss_pred_u0 = mseloss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
    l2loss_pred_u0 = l2loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
    l3loss_pred_u0 = l3loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()

    fmid = pred_u0.shape[1] // 4
    pred_u0 = pred_u0.squeeze(-1)
    y = y.squeeze(-1)

    fftmseloss_pred_u0 = fftmseloss_fn(pred_u0, y).item()
    fftmseloss_low_pred_u0 = fftmseloss_fn(pred_u0, y, 0, fmid).item()
    fftmseloss_mid_pred_u0 = fftmseloss_fn(pred_u0, y, fmid, 2 * fmid).item()
    fftmseloss_hi_pred_u0 = fftmseloss_fn(pred_u0, y, 2 * fmid).item()

    fftl2loss_pred_u0 = fftl2loss_fn(pred_u0, y).item()
    fftl2loss_low_pred_u0 = fftl2loss_fn(pred_u0, y, 0, fmid).item()
    fftl2loss_mid_pred_u0 = fftl2loss_fn(pred_u0, y, fmid, 2 * fmid).item()
    fftl2loss_hi_pred_u0 = fftl2loss_fn(pred_u0, y, 2 * fmid).item()

    fftl3loss_pred_u0 = fftl3loss_fn(pred_u0, y).item()
    fftl3loss_low_pred_u0 = fftl3loss_fn(pred_u0, y, 0, fmid).item()
    fftl3loss_mid_pred_u0 = fftl3loss_fn(pred_u0, y, fmid, 2 * fmid).item()
    fftl3loss_hi_pred_u0 = fftl3loss_fn(pred_u0, y, 2 * fmid).item()

    return {
        "mseloss_u0": mseloss_u0,
        "l2loss_u0": l2loss_u0,
        "l3loss_u0": l3loss_u0,
        "mseloss_pred_u0": mseloss_pred_u0,
        "l2loss_pred_u0": l2loss_pred_u0,
        "l3loss_pred_u0": l3loss_pred_u0,
        "fftmseloss_u0": fftmseloss_u0,
        "fftmseloss_low_u0": fftmseloss_low_u0,
        "fftmseloss_mid_u0": fftmseloss_mid_u0,
        "fftmseloss_hi_u0": fftmseloss_hi_u0,
        "fftmseloss_pred_u0": fftmseloss_pred_u0,
        "fftmseloss_low_pred_u0": fftmseloss_low_pred_u0,
        "fftmseloss_mid_pred_u0": fftmseloss_mid_pred_u0,
        "fftmseloss_hi_pred_u0": fftmseloss_hi_pred_u0,
        "fftl2loss_u0": fftl2loss_u0,
        "fftl2loss_low_u0": fftl2loss_low_u0,
        "fftl2loss_mid_u0": fftl2loss_mid_u0,
        "fftl2loss_hi_u0": fftl2loss_hi_u0,
        "fftl2loss_pred_u0": fftl2loss_pred_u0,
        "fftl2loss_low_pred_u0": fftl2loss_low_pred_u0,
        "fftl2loss_mid_pred_u0": fftl2loss_mid_pred_u0,
        "fftl2loss_hi_pred_u0": fftl2loss_hi_pred_u0,
        "fftl3loss_u0": fftl3loss_u0,
        "fftl3loss_low_u0": fftl3loss_low_u0,
        "fftl3loss_mid_u0": fftl3loss_mid_u0,
        "fftl3loss_hi_u0": fftl3loss_hi_u0,
        "fftl3loss_pred_u0": fftl3loss_pred_u0,
        "fftl3loss_low_pred_u0": fftl3loss_low_pred_u0,
        "fftl3loss_mid_pred_u0": fftl3loss_mid_pred_u0,
        "fftl3loss_hi_pred_u0": fftl3loss_hi_pred_u0,
    }
