"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     train_models_forward.py
  Authors:  Makoto Takamoto (makoto.takamoto@neclab.eu)
            Timothy Praditia (timothy.praditia@iws.uni-stuttgart.de)
            Raphael Leiteritz (raphael.leiteritz@ipvs.uni-stuttgart.de)

NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from pdebench.models.metrics import predict_time_benchmark, save_prediction_results

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_benchmark_FNO(
    num_workers,
    modes,
    width,
    initial_step,
    prediction_step,
    num_channels,
    batch_size,
    flnm,
    single_file,
    base_path,
    model_save_path,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    target_frames=100,
    warmup_runs=5,
    average_runs=10,
):
    """Load FNO model and run predict_time_benchmark."""
    from pdebench.models.fno.fno import FNO1d, FNO2d, FNO3d
    from pdebench.models.fno.utils import FNODatasetMult, FNODatasetSingle

    if single_file:
        model_name = flnm[:-5] + "_FNO"
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
        val_data = FNODatasetMult(
            flnm,
            initial_step=initial_step,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            if_test=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    _, _data, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
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

    model_path = Path(model_save_path) / "FNO" / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please check if the model has been trained and saved at this path."
        )
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    data_name = os.path.splitext(flnm)[0]
    avg_time_per_frame = predict_time_benchmark(
        val_loader=val_loader,
        model=model,
        mode="FNO",
        initial_step=initial_step,
        prediction_step=prediction_step,
        target_frames=target_frames,
        warmup_runs=warmup_runs,
        average_runs=average_runs,
        data_name=data_name,
    )
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

    return avg_time_per_frame


def run_benchmark_Unet(
    num_workers,
    initial_step,
    prediction_step,
    in_channels,
    out_channels,
    batch_size,
    flnm,
    single_file,
    base_path,
    model_save_path,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    training_type,
    ar_mode=None,
    pushforward=None,
    unroll_step=None,
    target_frames=100,
    warmup_runs=5,
    average_runs=10,
):
    """Load Unet model and run predict_time_benchmark."""
    from pdebench.models.unet.unet import UNet1d, UNet2d, UNet3d
    from pdebench.models.unet.utils import UNetDatasetMult, UNetDatasetSingle

    if single_file:
        model_name = flnm[:-5] + "_Unet"
        val_data = UNetDatasetSingle(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            if_test=True,
            saved_folder=base_path,
            initial_step=initial_step,
        )
    else:
        model_name = flnm + "_Unet"
        val_data = UNetDatasetMult(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            if_test=True,
            saved_folder=base_path,
            initial_step=initial_step,
        )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    _, _data = next(iter(val_loader))
    dimensions = len(_data.shape)

    if training_type in ["autoregressive"]:
        if ar_mode:
            if pushforward and unroll_step is not None:
                model_name = model_name + "-PF-" + str(unroll_step)
            elif not pushforward:
                model_name = model_name + "-AR"
        else:
            model_name = model_name + "-1-step"

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

    model_path = Path(model_save_path) / "Unet" / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please check if the model has been trained and saved at this path."
        )
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    data_name = os.path.splitext(flnm)[0]
    avg_time_per_frame = predict_time_benchmark(
        val_loader=val_loader,
        model=model,
        mode="Unet",
        initial_step=initial_step,
        prediction_step=prediction_step,
        target_frames=target_frames,
        warmup_runs=warmup_runs,
        average_runs=average_runs,
        data_name=data_name,
    )
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

    return avg_time_per_frame


def run_benchmark_Transolver(
    num_workers,
    n_hidden,
    n_layers,
    n_head,
    slice_num,
    mlp_ratio,
    initial_step,
    prediction_step,
    num_channels,
    batch_size,
    flnm,
    single_file,
    base_path,
    model_save_path,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    target_frames=100,
    warmup_runs=5,
    average_runs=10,
    dropout=0.0,
    act="gelu",
    kernel=3,
):
    """Load Transolver model and run predict_time_benchmark."""
    from pdebench.models.fno.utils import FNODatasetMult, FNODatasetSingle
    from pdebench.models.transolver.transolver import (
        Transolver1d,
        Transolver2d,
        Transolver3d,
    )

    if single_file:
        model_name = flnm[:-5] + "_Transolver"
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
        model_name = flnm + "_Transolver"
        val_data = FNODatasetMult(
            flnm,
            initial_step=initial_step,
            saved_folder=base_path,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            if_test=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    _, _data, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    if dimensions == 4:
        model = Transolver1d(
            num_channels=num_channels,
            initial_step=initial_step,
            prediction_step=prediction_step,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_head=n_head,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            act=act,
        ).to(device)
    elif dimensions == 5:
        H, W = _data.shape[1], _data.shape[2]
        model = Transolver2d(
            num_channels=num_channels,
            initial_step=initial_step,
            prediction_step=prediction_step,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_head=n_head,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            act=act,
            kernel=kernel,
            H=H,
            W=W,
        ).to(device)
    elif dimensions == 6:
        H, W, D = _data.shape[1], _data.shape[2], _data.shape[3]
        model = Transolver3d(
            num_channels=num_channels,
            initial_step=initial_step,
            prediction_step=prediction_step,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_head=n_head,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            act=act,
            kernel=kernel,
            H=H,
            W=W,
            D=D,
        ).to(device)

    model_path = Path(model_save_path) / "Transolver" / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please check if the model has been trained and saved at this path."
        )
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    data_name = os.path.splitext(flnm)[0]
    avg_time_per_frame = predict_time_benchmark(
        val_loader=val_loader,
        model=model,
        mode="FNO",
        initial_step=initial_step,
        prediction_step=prediction_step,
        target_frames=target_frames,
        warmup_runs=warmup_runs,
        average_runs=average_runs,
        data_name=data_name,
    )
    save_path = save_prediction_results(
        val_loader=val_loader,
        model=model,
        mode="FNO",
        initial_step=initial_step,
        prediction_step=prediction_step,
        model_name="Transolver",
        dataset_name=data_name,
    )
    if save_path:
        logger.info(f"Prediction results saved to {save_path}")

    return avg_time_per_frame


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    if hasattr(cfg.args, "if_benchmark") and cfg.args.if_benchmark:
        target_frames = getattr(cfg.args, "target_frames", 100)
        warmup_runs = getattr(cfg.args, "warmup_runs", 5)
        average_runs = getattr(cfg.args, "average_runs", 10)

        if cfg.args.model_name == "FNO":
            logger.info("Running FNO benchmark...")
            avg_time = run_benchmark_FNO(
                num_workers=cfg.args.num_workers,
                modes=cfg.args.modes,
                width=cfg.args.width,
                initial_step=cfg.args.initial_step,
                prediction_step=cfg.args.prediction_step,
                num_channels=cfg.args.num_channels,
                batch_size=cfg.args.batch_size,
                flnm=cfg.args.filename,
                single_file=cfg.args.single_file,
                base_path=cfg.args.data_path,
                model_save_path=cfg.args.model_save_path,
                reduced_resolution=cfg.args.reduced_resolution,
                reduced_resolution_t=cfg.args.reduced_resolution_t,
                reduced_batch=cfg.args.reduced_batch,
                target_frames=target_frames,
                warmup_runs=warmup_runs,
                average_runs=average_runs,
            )
            logger.info(
                f"FNO benchmark completed. Average time per frame: {avg_time:.5f}s"
            )
            return

        elif cfg.args.model_name == "Unet":
            logger.info("Running Unet benchmark...")
            avg_time = run_benchmark_Unet(
                num_workers=cfg.args.num_workers,
                initial_step=cfg.args.initial_step,
                prediction_step=cfg.args.prediction_step,
                in_channels=cfg.args.in_channels,
                out_channels=cfg.args.out_channels,
                batch_size=cfg.args.batch_size,
                flnm=cfg.args.filename,
                single_file=cfg.args.single_file,
                base_path=cfg.args.data_path,
                model_save_path=cfg.args.model_save_path,
                reduced_resolution=cfg.args.reduced_resolution,
                reduced_resolution_t=cfg.args.reduced_resolution_t,
                reduced_batch=cfg.args.reduced_batch,
                training_type=cfg.args.training_type,
                ar_mode=getattr(cfg.args, "ar_mode", None),
                pushforward=getattr(cfg.args, "pushforward", None),
                unroll_step=getattr(cfg.args, "unroll_step", None),
                target_frames=target_frames,
                warmup_runs=warmup_runs,
                average_runs=average_runs,
            )
            logger.info(
                f"Unet benchmark completed. Average time per frame: {avg_time:.5f}s"
            )
            return

        elif cfg.args.model_name == "Transolver":
            logger.info("Running Transolver benchmark...")
            avg_time = run_benchmark_Transolver(
                num_workers=cfg.args.num_workers,
                n_hidden=cfg.args.n_hidden,
                n_layers=cfg.args.n_layers,
                n_head=cfg.args.n_head,
                slice_num=cfg.args.slice_num,
                mlp_ratio=cfg.args.mlp_ratio,
                initial_step=cfg.args.initial_step,
                prediction_step=getattr(cfg.args, "prediction_step", 1),
                num_channels=cfg.args.num_channels,
                batch_size=cfg.args.batch_size,
                flnm=cfg.args.filename,
                single_file=cfg.args.single_file,
                base_path=cfg.args.data_path,
                model_save_path=cfg.args.model_save_path,
                reduced_resolution=cfg.args.reduced_resolution,
                reduced_resolution_t=cfg.args.reduced_resolution_t,
                reduced_batch=cfg.args.reduced_batch,
                target_frames=target_frames,
                warmup_runs=warmup_runs,
                average_runs=average_runs,
                dropout=getattr(cfg.args, "dropout", 0.0),
                act=getattr(cfg.args, "act", "gelu"),
                kernel=getattr(cfg.args, "kernel", 3),
            )
            logger.info(
                f"Transolver benchmark completed. Average time per frame: {avg_time:.5f}s"
            )
            return

        else:
            logger.error(f"Benchmark not supported for model: {cfg.args.model_name}")
            return

    if cfg.args.model_name == "FNO":
        from pdebench.models.fno.train import run_training as run_training_FNO

        logger.info("FNO")
        run_training_FNO(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            num_workers=cfg.args.num_workers,
            modes=cfg.args.modes,
            width=cfg.args.width,
            initial_step=cfg.args.initial_step,
            prediction_step=cfg.args.prediction_step,
            t_train=cfg.args.t_train,
            training_type=cfg.args.training_type,
            num_channels=cfg.args.num_channels,
            batch_size=cfg.args.batch_size,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            base_path=cfg.args.data_path,
            model_save_path=cfg.args.model_save_path,
            result_save_path=cfg.args.result_save_path,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
        )
    elif cfg.args.model_name == "Unet":
        from pdebench.models.unet.train import run_training as run_training_Unet

        logger.info("Unet")
        run_training_Unet(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            num_workers=cfg.args.num_workers,
            initial_step=cfg.args.initial_step,
            prediction_step=cfg.args.prediction_step,
            t_train=cfg.args.t_train,
            in_channels=cfg.args.in_channels,
            out_channels=cfg.args.out_channels,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            batch_size=cfg.args.batch_size,
            unroll_step=cfg.args.unroll_step,
            ar_mode=cfg.args.ar_mode,
            pushforward=cfg.args.pushforward,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            base_path=cfg.args.data_path,
            model_save_path=cfg.args.model_save_path,
            result_save_path=cfg.args.result_save_path,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
        )
    elif cfg.args.model_name == "PINN":
        from pdebench.models.pinn.train import run_training as run_training_PINN

        logger.info("PINN")
        run_training_PINN(
            scenario=cfg.args.scenario,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            seed=cfg.args.seed,
            input_ch=cfg.args.input_ch,
            output_ch=cfg.args.output_ch,
            root_path=cfg.args.root_path,
            val_num=cfg.args.val_num,
            if_periodic_bc=cfg.args.if_periodic_bc,
            aux_params=cfg.args.aux_params,
        )
    elif cfg.args.model_name == "Transolver":
        from pdebench.models.transolver.train import (
            run_training as run_training_Transolver,
        )

        logger.info("Transolver")
        run_training_Transolver(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            num_workers=cfg.args.num_workers,
            n_hidden=cfg.args.n_hidden,
            n_layers=cfg.args.n_layers,
            n_head=cfg.args.n_head,
            slice_num=cfg.args.slice_num,
            mlp_ratio=cfg.args.mlp_ratio,
            initial_step=cfg.args.initial_step,
            prediction_step=getattr(cfg.args, "prediction_step", 1),
            t_train=cfg.args.t_train,
            training_type=cfg.args.training_type,
            num_channels=cfg.args.num_channels,
            batch_size=cfg.args.batch_size,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            base_path=cfg.args.data_path,
            model_save_path=cfg.args.model_save_path,
            result_save_path=cfg.args.result_save_path,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
            dropout=getattr(cfg.args, "dropout", 0.0),
            act=getattr(cfg.args, "act", "gelu"),
            kernel=getattr(cfg.args, "kernel", 3),
        )


if __name__ == "__main__":
    main()
