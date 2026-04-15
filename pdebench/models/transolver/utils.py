"""
Data utilities for Transolver.

Reuses the FNO dataset classes since both models consume the same
(xx, yy, grid) tuple format from HDF5 data.
"""

from pdebench.models.fno.utils import FNODatasetMult, FNODatasetSingle

TransolverDatasetSingle = FNODatasetSingle
TransolverDatasetMult = FNODatasetMult

__all__ = [
    "FNODatasetSingle",
    "FNODatasetMult",
    "TransolverDatasetSingle",
    "TransolverDatasetMult",
]
