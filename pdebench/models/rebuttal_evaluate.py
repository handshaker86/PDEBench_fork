"""Periodic-grid metrics for PDEBench rebuttal predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _frame_number(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def load_velocity_npz(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if {"u", "v"} <= set(archive.files):
            return np.stack([archive["u"], archive["v"]], axis=1).astype(np.float64)
        frames = []
        for key in sorted(archive.files, key=_frame_number):
            frame = np.asarray(archive[key])
            if frame.shape[0] != 2:
                raise ValueError(f"{key} must contain (u, v), got {frame.shape}")
            frames.append(frame)
    return np.asarray(frames, dtype=np.float64)


def _periodic_derivative(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    if spacing <= 0:
        raise ValueError("grid spacing must be positive")
    return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (
        2 * spacing
    )


def divergence(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return _periodic_derivative(field[:, 0], dx, -1) + _periodic_derivative(
        field[:, 1], dy, -2
    )


def vorticity(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return _periodic_derivative(field[:, 1], dx, -1) - _periodic_derivative(
        field[:, 0], dy, -2
    )


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def metrics(target: np.ndarray, prediction: np.ndarray, dx: float, dy: float) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 4 or target.shape[1] != 2:
        raise ValueError(
            f"expected matching (frames, 2, h, w), got {target.shape} and {prediction.shape}"
        )
    target_div = divergence(target, dx, dy)
    prediction_div = divergence(prediction, dx, dy)
    target_vort = vorticity(target, dx, dy)
    prediction_vort = vorticity(prediction, dx, dy)
    epsilon = 1e-12
    spectrum_delta = np.log(np.abs(np.fft.fft2(target)) + 1e-8) - np.log(
        np.abs(np.fft.fft2(prediction)) + 1e-8
    )
    return {
        "rmse": _rms(prediction - target),
        "relative_vorticity_error": float(
            np.linalg.norm(prediction_vort - target_vort)
            / (np.linalg.norm(target_vort) + epsilon)
        ),
        "spectral_error": _rms(spectrum_delta),
        "d_gt": _rms(target_div),
        "d_pred": _rms(prediction_div),
        "e_div": _rms(prediction_div - target_div),
        "d_ratio": _rms(prediction_div) / (_rms(target_div) + epsilon),
    }


def continuity_residual(
    density_current: np.ndarray,
    density_next: np.ndarray,
    velocity_next: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Return d(rho)/dt + div(rho*u) on a periodic grid."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    density_current = np.asarray(density_current, dtype=np.float64)
    density_next = np.asarray(density_next, dtype=np.float64)
    velocity_next = np.asarray(velocity_next, dtype=np.float64)
    if (
        density_current.shape != density_next.shape
        or density_next.shape != velocity_next[:, 0].shape
    ):
        raise ValueError("density and velocity shapes do not align")
    mass_flux = velocity_next * density_next[:, None]
    return (density_next - density_current) / dt + divergence(mass_flux, dx, dy)


def evaluate(target, prediction, dx, dy):
    aggregate = metrics(target, prediction, dx, dy)
    rows = [
        {
            "frame": frame,
            **metrics(target[frame : frame + 1], prediction[frame : frame + 1], dx, dy),
        }
        for frame in range(target.shape[0])
    ]
    return aggregate, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dy", type=float, required=True)
    args = parser.parse_args()

    target = load_velocity_npz(args.target)
    prediction = load_velocity_npz(args.prediction)
    aggregate, rows = evaluate(target, prediction, args.dx, args.dy)
    aggregate.update(
        {
            "dx": args.dx,
            "dy": args.dy,
            "boundary": "periodic-central",
            "num_frames": len(rows),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rebuttal_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "rebuttal_metrics_per_frame.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
