"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     utils.py
  Authors:  Timothy Praditia (timothy.praditia@iws.uni-stuttgart.de)
            Raphael Leiteritz (raphael.leiteritz@ipvs.uni-stuttgart.de)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Francesco Alesiani (makoto.takamoto@neclab.eu)

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

import math as mt
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class UNetDatasetSingle(Dataset):
    def __init__(
        self,
        filename,
        initial_step=10,
        saved_folder="../data/",
        reduced_resolution=1,
        reduced_resolution_t=1,
        reduced_batch=1,
        if_test=False,
        test_ratio=0.01,
        num_samples_max=-1,
    ):
        super().__init__()
        self.h5_path = Path(saved_folder + filename).resolve()
        assert self.h5_path.exists(), f"HDF5 file not found at: {self.h5_path}"

        assert self.h5_path.suffix != ".h5", "HDF5 data is assumed!!"

        self.initial_step = initial_step
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t
        self.reduced_batch = reduced_batch
        self.file_handle = None

        with h5py.File(self.h5_path, "r") as f:
            keys = list(f.keys())
            self.is_scalar = "tensor" in keys

            if self.is_scalar:
                dset_name = "tensor"
                shape = f[dset_name].shape
                self.dim = len(shape) - 2  # (batch, time, x, y, ...)
                if self.dim == 2:
                    self.keys_to_load = ["nu", "tensor"]
                else:
                    self.keys_to_load = ["tensor"]
            else:
                dset_name = "density"
                shape = f[dset_name].shape
                self.dim = len(shape) - 2
                if self.dim == 1:
                    self.keys_to_load = ["density", "pressure", "Vx"]
                elif self.dim == 2:
                    self.keys_to_load = ["density", "pressure", "Vx", "Vy"]
                elif self.dim == 3:
                    self.keys_to_load = ["density", "pressure", "Vx", "Vy", "Vz"]
                else:
                    raise ValueError(f"Unsupported dimension: {self.dim}")

            self.original_shape = shape

        num_samples_total = self.original_shape[0]

        num_samples_reduced = num_samples_total // self.reduced_batch

        if num_samples_max > 0:
            num_samples = min(num_samples_max, num_samples_reduced)
        else:
            num_samples = num_samples_reduced

        test_idx = int(num_samples * (1 - test_ratio))
        all_indices_reduced = list(range(num_samples))

        if if_test:
            self.indices = all_indices_reduced[test_idx:]
        else:
            self.indices = all_indices_reduced[:test_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file_handle is None:
            self.file_handle = h5py.File(self.h5_path, "r")

        reduced_idx = self.indices[idx]
        h5_idx = reduced_idx * self.reduced_batch
        rr = self.reduced_resolution
        rrt = self.reduced_resolution_t

        item_data = self._load_item(h5_idx, rrt, rr)
        xx_data = item_data[..., : self.initial_step, :]
        yy_data = item_data

        return xx_data, yy_data

    def _load_item(self, h5_idx, rrt, rr):
        slicer_spatial = [slice(None, None, rr)] * self.dim
        slicer_time = slice(None, None, rrt)

        if not self.is_scalar:
            slicer = tuple([h5_idx, slicer_time] + slicer_spatial)

            # (t, x, y, ...) -> (x, y, ..., t)
            transpose_order = (*range(1, self.dim + 1), 0)

            channels = []
            for key in self.keys_to_load:
                _data = self.file_handle[key][slicer]
                _data = np.transpose(_data, transpose_order)
                channels.append(_data)

            # (x, y, ..., t, Channels)
            return np.stack(channels, axis=-1)

        else:  # Scalar equations
            if self.dim == 1:
                slicer = (h5_idx, slicer_time, slicer_spatial[0])
                _data = self.file_handle["tensor"][slicer]  # (t, x)
                _data = np.transpose(_data, (1, 0))  # (x, t)
                return _data[..., None]  # (x, t, 1)

            elif self.dim == 2:  # 2D Darcy flow
                _data_u = self.file_handle["tensor"][h5_idx, :, ::rr, ::rr]  # (t, x, y)
                _data_u = np.transpose(_data_u, (1, 2, 0))  # (x, y, t)

                _data_nu = self.file_handle["nu"][h5_idx, None, ::rr, ::rr]  # (1, x, y)
                _data_nu = np.transpose(_data_nu, (1, 2, 0))  # (x, y, 1)
                _data_nu = np.tile(_data_nu, (1, 1, _data_u.shape[-1]))  # (x, y, t)

                # (x, y, t, ch=2)
                data = np.stack([_data_nu, _data_u], axis=-1)
                return data

            elif self.dim == 3:
                slicer = (h5_idx, slicer_time, *slicer_spatial)
                _data = self.file_handle["tensor"][slicer]  # (t, x, y, z)
                _data = np.transpose(_data, (1, 2, 3, 0))  # (x, y, z, t)
                return _data[..., None]  # (x, y, z, t, 1)

            else:
                raise ValueError(f"Unsupported scalar dimension: {self.dim}")

    def __del__(self):
        if hasattr(self, "file_handle") and self.file_handle:
            self.file_handle.close()
            self.file_handle = None


class UNetDatasetMult(Dataset):
    def __init__(
        self,
        filename,
        initial_step=10,
        saved_folder="../data/",
        reduced_resolution=1,
        reduced_resolution_t=1,
        reduced_batch=1,
        if_test=False,
        test_ratio=0.01,
    ):
        """

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """

        # Define path to files
        self.file_path = Path(saved_folder + "/" + filename + ".h5").resolve()

        # Extract list of seeds
        with h5py.File(self.file_path, "r") as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])

        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Open file and read data
        with h5py.File(self.file_path, "r") as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype="f")
            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape) - 1))
            permute_idx.extend([0, -1])
            data = data.permute(permute_idx)

        return data[..., : self.initial_step, :], data
