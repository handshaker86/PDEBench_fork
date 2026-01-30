from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
from pdebench.models.fno.fno import FNO1d, FNO2d, FNO3d
from pdebench.models.fno.utils import FNODatasetMult, FNODatasetSingle
from pdebench.models.metrics import metrics, save_prediction_results
from pdebench.models.training_logger import TrainingLogger
from torch import nn

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(
    if_training,
    continue_training,
    num_workers,
    modes,
    width,
    initial_step,
    prediction_step,
    t_train,
    num_channels,
    batch_size,
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
        model_name = flnm[:-5] + "_FNO"
        result_save_path = result_save_path + "/FNO/" + flnm + "/"
        print("FNODatasetSingle")
        train_data = FNODatasetSingle(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
            saved_folder=base_path,
            if_test=False,
        )
        val_data = FNODatasetSingle(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
            if_test=True,
            saved_folder=base_path,
        )
    else:
        model_name = flnm + "_FNO"
        result_save_path = result_save_path + "/FNO/" + flnm + "/"
        print("FNODatasetMult")
        train_data = FNODatasetMult(
            flnm,
            initial_step=initial_step,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            if_test=False,
        )
        val_data = FNODatasetMult(
            flnm,
            initial_step=initial_step,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            if_test=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    _, _data, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print("Spatial Dimension", dimensions - 3)
    if dimensions == 4:
        model = FNO1d(
            num_channels=num_channels,
            width=width,
            modes=modes,
            initial_step=initial_step,
            prediction_step=prediction_step,
        ).to(device)
    elif dimensions == 5:
        model = FNO2d(
            num_channels=num_channels,
            width=width,
            modes1=modes,
            modes2=modes,
            initial_step=initial_step,
            prediction_step=prediction_step,
        ).to(device)
    elif dimensions == 6:
        model = FNO3d(
            num_channels=num_channels,
            width=width,
            modes1=modes,
            modes2=modes,
            modes3=modes,
            initial_step=initial_step,
            prediction_step=prediction_step,
        ).to(device)

    t_train = min(t_train, _data.shape[-2])
    model_path = model_save_path + "FNO/" + model_name + ".pt"
    Path(model_save_path + "/FNO/").mkdir(parents=True, exist_ok=True)

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
            initial_step=initial_step,
            prediction_step=prediction_step,
            result_save_path=result_save_path,
            rollout_plot=True,
        )
        with Path(model_name + ".pickle").open("wb") as pb:
            pickle.dump(errs, pb)
        data_name = os.path.splitext(flnm)[0] if single_file else flnm
        save_path = save_prediction_results(
            val_loader=val_loader,
            model=model,
            mode="FNO",
            initial_step=initial_step,
            prediction_step=prediction_step,
            model_name="FNO",
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

    for ep in range(start_epoch, epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy, grid in train_loader:
            loss = 0
            xx = xx.to(device)  # noqa: PLW2901
            yy = yy.to(device)  # noqa: PLW2901
            grid = grid.to(device)  # noqa: PLW2901
            pred = yy[..., :initial_step, :]
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)

            if training_type in ["autoregressive"]:
                for t in range(
                    initial_step, t_train - prediction_step + 1, prediction_step
                ):
                    inp = xx.reshape(inp_shape)
                    y = yy[..., t : t + prediction_step, :]
                    im = model(inp, grid)
                    _batch = im.size(0)
                    loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
                    pred = torch.cat((pred, im), -2)
                    xx = torch.cat(
                        (xx[..., prediction_step:, :], im), dim=-2
                    )  # noqa: PLW2901

                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy[..., :t_train, :]  # if t_train is not -1

                pred_len = pred.shape[-2]
                l2_full = loss_fn(
                    pred.reshape(_batch, -1), _yy[..., :pred_len, :].reshape(_batch, -1)
                )
                train_l2_full += l2_full.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if training_type in ["single"]:
                x = xx[..., 0, :]
                y = yy[..., t_train - prediction_step : t_train, :]
                pred = model(x, grid)
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
                for xx, yy, grid in val_loader:
                    loss = 0
                    xx = xx.to(device)  # noqa: PLW2901
                    yy = yy.to(device)  # noqa: PLW2901
                    grid = grid.to(device)  # noqa: PLW2901

                    if training_type in ["autoregressive"]:
                        pred = yy[..., :initial_step, :]
                        inp_shape = list(xx.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)

                        T = yy.shape[-2]
                        for t in range(
                            initial_step, T - prediction_step + 1, prediction_step
                        ):
                            inp = xx.reshape(inp_shape)
                            y = yy[..., t : t + prediction_step, :]
                            im = model(inp, grid)
                            _batch = im.size(0)
                            loss += loss_fn(
                                im.reshape(_batch, -1), y.reshape(_batch, -1)
                            )

                            pred = torch.cat((pred, im), -2)

                            xx = torch.cat(
                                (xx[..., prediction_step:, :], im), dim=-2
                            )  # noqa: PLW2901

                        val_l2_step += loss.item()
                        _batch = yy.size(0)

                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy[..., initial_step:t_train, :]

                        pred_len = _pred.shape[-2]
                        val_l2_full += loss_fn(
                            _pred.reshape(_batch, -1),
                            _yy[..., :pred_len, :].reshape(_batch, -1),
                        ).item()

                    if training_type in ["single"]:
                        x = xx[..., 0, :]
                        y = yy[..., t_train - prediction_step : t_train, :]
                        pred = model(x, grid)
                        _batch = yy.size(0)
                        loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))

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

        ckpt_path = Path(model_save_path) / "FNO" / f"ckpt-{ep}.pt"
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
            f"trainL2: {train_l2_full:.5f}, testL2: {val_l2_full:.5f}"
        )

    logger_training.plot_loss_curves(save_plot=True)
    logger.info(
        f"Training done, loss history saved to {logger_training.loss_history_file}"
    )


if __name__ == "__main__":
    run_training()
