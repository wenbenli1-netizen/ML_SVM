from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a NIfTI MRI file and save simple slice views.")
    parser.add_argument("--path", required=True, help="Path to a .nii.gz file.")
    parser.add_argument(
        "--output-dir",
        default="results/read_example",
        help="Directory used to save slice figures.",
    )
    return parser.parse_args()


def save_slice_views(data: np.ndarray, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cx, cy, cz = (dim // 2 for dim in data.shape)
    views = [
        ("sagittal", np.rot90(data[cx, :, :])),
        ("coronal", np.rot90(data[:, cy, :])),
        ("axial", np.rot90(data[:, :, cz])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, image) in zip(axes, views):
        ax.imshow(image, cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    save_path = output_dir / f"{stem}_three_views.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"saved figure: {save_path}")


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    output_dir = Path(args.output_dir)

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    nonzero = data[data != 0]

    print(f"path: {path}")
    print(f"shape: {data.shape}")
    print(f"dtype: {img.get_data_dtype()}")
    print(f"zooms: {tuple(float(z) for z in img.header.get_zooms()[:3])}")
    print("affine:")
    print(img.affine)
    print(f"min: {float(data.min()):.4f}")
    print(f"max: {float(data.max()):.4f}")
    print(f"mean: {float(data.mean()):.4f}")
    print(f"std: {float(data.std()):.4f}")
    print(f"zero_ratio: {float((data == 0).mean()):.4f}")
    if nonzero.size:
        print(f"nonzero_p95: {float(np.percentile(nonzero, 95)):.4f}")
        print(f"nonzero_p99: {float(np.percentile(nonzero, 99)):.4f}")

    save_slice_views(data, output_dir, path.stem.replace(".nii", ""))


if __name__ == "__main__":
    main()
