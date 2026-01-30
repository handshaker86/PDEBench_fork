from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
from pdebench.models.metrics import metrics, save_prediction_results
from pdebench.models.training_logger import TrainingLogger
from pdebench.models.unet.unet import UNet1d, UNet2d, UNet3d
from pdebench.models.unet.utils import UNetDatasetMult, UNetDatasetSingle
from torch import nn

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(
    if_training,
    continue_training,
    num_workers,
    initial_step,
    t_train,
    prediction_step,
    in_channels,
    out_channels,
    batch_size,
    unroll_step,
    ar_mode,
    pushforward,
    epochs,
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    model_update,
    flnm,
    single_file,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    plot,
    channel_plot,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max,
    base_path="../data/",
    model_save_path="models/",
    result_save_path="results/",
    training_type="autoregressive",
):
    print(
        f"Epochs = {epochs}, lr = {learning_rate}, scheduler step = {scheduler_step}, gamma = {scheduler_gamma}"
    )

    if single_file:
        model_name = flnm[:-5] + "_Unet"
        result_save_path = result_save_path + "/Unet/" + flnm + "/"
        train_data = UNetDatasetSingle(
            flnm,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
        )
        val_data = UNetDatasetSingle(
            flnm,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
            if_test=True,
        )
    else:
        model_name = flnm + "_Unet"
        result_save_path = result_save_path + "/Unet/" + flnm + "/"
        train_data = UNetDatasetMult(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            saved_folder=base_path,
            initial_step=initial_step,
        )
        val_data = UNetDatasetMult(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            if_test=True,
            saved_folder=base_path,
            initial_step=initial_step,
        )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    _, _data = next(iter(val_loader))
    dimensions = len(_data.shape)
    print(f"Spatial dimension: {dimensions - 3}")
    if training_type in ["autoregressive"]:
        if dimensions == 4:
            model = UNet1d(
                in_channels * initial_step,
                out_channels,
                prediction_step=prediction_step,
            ).to(device)
        elif dimensions == 5:
            model = UNet2d(
                in_channels * initial_step,
                out_channels,
                prediction_step=prediction_step,
            ).to(device)
        elif dimensions == 6:
            model = UNet3d(
                in_channels * initial_step,
                out_channels,
                prediction_step=prediction_step,
            ).to(device)
    if training_type in ["single"]:
        if dimensions == 4:
            model = UNet1d(
                in_channels, out_channels, prediction_step=prediction_step
            ).to(device)
        elif dimensions == 5:
            model = UNet2d(
                in_channels, out_channels, prediction_step=prediction_step
            ).to(device)
        elif dimensions == 6:
            model = UNet3d(
                in_channels, out_channels, prediction_step=prediction_step
            ).to(device)

    t_train = min(t_train, _data.shape[-2])
    if t_train - unroll_step < 1:
        unroll_step = t_train - 1

    if training_type in ["autoregressive"]:
        if ar_mode:
            if pushforward:
                model_name = model_name + "-PF-" + str(unroll_step)
            if not pushforward:
                unroll_step = _data.shape[-2]
                model_name = model_name + "-AR"
        else:
            model_name = model_name + "-1-step"

    model_path = Path(model_save_path) / "Unet" / f"{model_name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters = {total_params}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.inf
    start_epoch = 0

    logger_training = TrainingLogger(result_save_path, model_name)
    logger_training.load_history()

    if not if_training:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        errs = metrics(
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
            mode="Unet",
            initial_step=initial_step,
            prediction_step=prediction_step,
            result_save_path=result_save_path,
            rollout_plot=True,
        )

        path = Path(model_name + ".pickle")
        with path.open("wb") as f:
            pickle.dump(errs, f)
        data_name = os.path.splitext(flnm)[0] if single_file else flnm
        save_path = save_prediction_results(
            val_loader=val_loader,
            model=model,
            mode="Unet",
            initial_step=initial_step,
            prediction_step=prediction_step,
            model_name="Unet",
            dataset_name=data_name,
        )
        if save_path:
            logger.info(f"Prediction results saved to {save_path}")

        return

    if continue_training:
        print("Restoring model from checkpoint...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.train()

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint["epoch"]
        loss_val_min = checkpoint["loss"]
        last_recorded_epoch = logger_training.get_last_epoch()
        if last_recorded_epoch < start_epoch:
            start_epoch = max(start_epoch, last_recorded_epoch + 1)

    print("Start training...")

    if ar_mode:
        for ep in range(start_epoch, epochs):
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0

            for xx, yy in train_loader:
                loss = 0
                xx_tensor = xx.to(device)
                yy_tensor = yy.to(device)

                if training_type in ["autoregressive"]:
                    pred = yy_tensor[..., :initial_step, :]
                    inp_shape = list(xx_tensor.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    for t in range(
                        initial_step, t_train - prediction_step + 1, prediction_step
                    ):
                        if t < t_train - unroll_step:
                            with torch.no_grad():
                                inp = xx_tensor.reshape(inp_shape)
                                temp_shape = [0, -1]
                                temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                                inp = inp.permute(temp_shape)
                                im = model(inp)
                                pred = torch.cat((pred, im), -2)
                                xx_tensor = torch.cat(
                                    (xx_tensor[..., prediction_step:, :], im),
                                    dim=-2,
                                )
                        else:
                            inp = xx_tensor.reshape(inp_shape)
                            temp_shape = [0, -1]
                            temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                            inp = inp.permute(temp_shape)
                            y = yy_tensor[..., t : t + prediction_step, :]
                            im = model(inp)
                            loss += loss_fn(
                                im.reshape(batch_size, -1), y.reshape(batch_size, -1)
                            )
                            pred = torch.cat((pred, im), -2)
                            xx_tensor = torch.cat(
                                (xx_tensor[..., prediction_step:, :], im), dim=-2
                            )

                    train_l2_step += loss.item()
                    _batch = yy_tensor.size(0)
                    _yy = yy_tensor[..., :t_train, :]

                    pred_len = pred.shape[-2]
                    l2_full = loss_fn(
                        pred.reshape(_batch, -1),
                        _yy[..., :pred_len, :].reshape(_batch, -1),
                    )
                    train_l2_full += l2_full.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if training_type in ["single"]:
                    x = xx[..., 0, :]
                    y = yy[..., t_train - prediction_step : t_train, :]
                    pred = model(x.permute([0, 2, 1]))
                    _batch = yy.size(0)
                    loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))

                    train_l2_step += loss.item()
                    train_l2_full += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if ep % model_update == 0:
                val_l2_step = 0
                val_l2_full = 0
                with torch.no_grad():
                    for xx, yy in val_loader:
                        loss = 0
                        xx_tensor = xx.to(device)
                        yy_tensor = yy.to(device)

                        if training_type in ["autoregressive"]:
                            pred = yy_tensor[..., :initial_step, :]
                            inp_shape = list(xx.shape)
                            inp_shape = inp_shape[:-2]
                            inp_shape.append(-1)

                            T_val = yy_tensor.shape[-2]
                            for t in range(
                                initial_step,
                                T_val - prediction_step + 1,
                                prediction_step,
                            ):
                                inp = xx_tensor.reshape(inp_shape)
                                temp_shape = [0, -1]
                                temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                                inp = inp.permute(temp_shape)
                                y = yy_tensor[..., t : t + prediction_step, :]
                                im = model(inp)
                                loss += loss_fn(
                                    im.reshape(batch_size, -1),
                                    y.reshape(batch_size, -1),
                                )
                                pred = torch.cat((pred, im), -2)
                                xx_tensor = torch.cat(
                                    (xx_tensor[..., prediction_step:, :], im),
                                    dim=-2,
                                )

                            val_l2_step += loss.item()
                            _batch = yy.size(0)

                            _pred = pred[..., initial_step:t_train, :]
                            _yy = yy_tensor[..., initial_step:t_train, :]

                            pred_len = _pred.shape[-2]
                            val_l2_full += loss_fn(
                                _pred.reshape(_batch, -1),
                                _yy[..., :pred_len, :].reshape(_batch, -1),
                            ).item()

                        if training_type in ["single"]:
                            x = xx[..., 0, :]
                            y = yy[..., t_train - prediction_step : t_train, :]
                            pred = model(x.permute([0, 2, 1]))
                            _batch = yy.size(0)
                            loss += loss_fn(
                                pred.reshape(_batch, -1), y.reshape(_batch, -1)
                            )

                            val_l2_step += loss.item()
                            val_l2_full += loss.item()

                        if val_l2_full < loss_val_min:
                            loss_val_min = val_l2_full
                            torch.save(
                                {
                                    "epoch": ep,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "loss": loss_val_min,
                                },
                                model_path,
                            )

            t2 = default_timer()
            epoch_time = t2 - t1
            scheduler.step()

            val_l2_step_for_log = val_l2_step if ep % model_update == 0 else None
            val_l2_full_for_log = val_l2_full if ep % model_update == 0 else None
            logger_training.record(
                epoch=ep,
                train_loss_step=train_l2_step,
                train_loss_full=train_l2_full,
                val_loss_step=val_l2_step_for_log,
                val_loss_full=val_l2_full_for_log,
                epoch_time=epoch_time,
            )
            logger_training.save()

            ckpt_path = model_path.parent / f"ckpt-{ep}.pt"
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_val_min,
                },
                ckpt_path,
            )
            if ep % 10 == 0:
                logger_training.plot_loss_curves(save_plot=True)
            print(
                f"epoch: {ep}, loss: {loss.item():.5f}, t2-t1: {t2 - t1:.5f}, "
                f"trainL2: {train_l2_step:.5f}, testL2: {val_l2_step:.5f}"
            )

    else:
        for ep in range(start_epoch, epochs):
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0

            for xx, yy in train_loader:
                loss = 0

                xx_tensor = xx.to(device)
                yy_tensor = yy.to(device)
                pred = yy_tensor[..., :initial_step, :]
                inp_shape = list(xx_tensor.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for t in range(
                    initial_step, t_train - prediction_step + 1, prediction_step
                ):
                    inp = yy_tensor[..., t - initial_step : t, :].reshape(inp_shape)
                    temp_shape = [0, -1]
                    temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                    inp = inp.permute(temp_shape)
                    inp = torch.normal(inp, 0.001)
                    y = yy_tensor[..., t : t + prediction_step, :]
                    im = model(inp)
                    loss += loss_fn(
                        im.reshape(batch_size, -1), y.reshape(batch_size, -1)
                    )
                    pred = torch.cat((pred, im), -2)

                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy_tensor[..., :t_train, :]  # if t_train is not -1

                pred_len = pred.shape[-2]
                l2_full = loss_fn(
                    pred.reshape(_batch, -1), _yy[..., :pred_len, :].reshape(_batch, -1)
                )
                train_l2_full += l2_full.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ep % model_update == 0 or ep == epochs:
                val_l2_step = 0
                val_l2_full = 0
                with torch.no_grad():
                    for xx, yy in val_loader:
                        loss = 0
                        xx_tensor = xx.to(device)
                        yy_tensor = yy.to(device)

                        pred = yy_tensor[..., :initial_step, :]
                        inp_shape = list(xx_tensor.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)

                        T_val = yy_tensor.shape[-2]
                        for t in range(
                            initial_step, T_val - prediction_step + 1, prediction_step
                        ):
                            inp = yy_tensor[..., t - initial_step : t, :].reshape(
                                inp_shape
                            )
                            temp_shape = [0, -1]
                            temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                            inp = inp.permute(temp_shape)

                            y = yy_tensor[..., t : t + prediction_step, :]

                            im = model(inp)

                            loss += loss_fn(
                                im.reshape(batch_size, -1), y.reshape(batch_size, -1)
                            )

                            pred = torch.cat((pred, im), -2)

                        val_l2_step += loss.item()
                        _batch = yy.size(0)

                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy_tensor[..., initial_step:t_train, :]

                        pred_len = _pred.shape[-2]
                        val_l2_full += loss_fn(
                            _pred.reshape(_batch, -1),
                            _yy[..., :pred_len, :].reshape(_batch, -1),
                        ).item()

                    if val_l2_full < loss_val_min:
                        loss_val_min = val_l2_full
                        torch.save(
                            {
                                "epoch": ep,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss_val_min,
                            },
                            model_path,
                        )

            t2 = default_timer()
            epoch_time = t2 - t1
            scheduler.step()

            val_l2_step_for_log = (
                val_l2_step if (ep % model_update == 0 or ep == epochs) else None
            )
            val_l2_full_for_log = (
                val_l2_full if (ep % model_update == 0 or ep == epochs) else None
            )
            logger_training.record(
                epoch=ep,
                train_loss_step=train_l2_step,
                train_loss_full=train_l2_full,
                val_loss_step=val_l2_step_for_log,
                val_loss_full=val_l2_full_for_log,
                epoch_time=epoch_time,
            )
            logger_training.save()

            ckpt_path = model_path.parent / f"ckpt-{ep}.pt"
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_val_min,
                },
                ckpt_path,
            )
            if ep % 10 == 0:
                logger_training.plot_loss_curves(save_plot=True)
            print(
                f"epoch: {ep}, loss: {loss.item():.5f}, t2-t1: {t2 - t1:.5f}, "
                f"trainL2: {train_l2_step:.5f}, testL2: {val_l2_step:.5f}"
            )

    logger_training.plot_loss_curves(save_plot=True)
    logger.info(
        f"Training done, loss history saved to {logger_training.loss_history_file}"
    )


if __name__ == "__main__":
    run_training()
