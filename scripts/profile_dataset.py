from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


FILENAME_PATTERN = re.compile(
    r"^(?P<source>[^+]+)\+(?P<subject>[^+]+)\+(?P<label>[A-Z]+)-m(?P<month>\d+)-(?P<scanner>.+?)\.nii\.gz$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the processed MRI dataset.")
    parser.add_argument("--data-dir", default="processed", help="Directory containing .nii.gz MRI files.")
    parser.add_argument("--output-dir", default="results/data_profile", help="Directory used to save summary tables.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def index_dataset(data_dir: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for path in sorted(data_dir.glob("*.nii.gz")):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        record = match.groupdict()
        record["path"] = str(path.resolve())
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No parsed .nii.gz files found in {data_dir}")
    df = pd.DataFrame(records)
    df["month"] = df["month"].astype(str).str.zfill(2)
    return df


def summarize_basic(df: pd.DataFrame, output_dir: Path) -> None:
    df.to_csv(output_dir / "dataset_index.csv", index=False)

    label_summary = (
        df.groupby("label")
        .agg(files=("path", "count"), subjects=("subject", "nunique"))
        .reset_index()
        .sort_values("label")
    )
    label_summary.to_csv(output_dir / "label_summary.csv", index=False)

    month_summary = df.groupby("month").size().reset_index(name="files").sort_values("month")
    month_summary["visit"] = month_summary["month"].apply(lambda value: f"m{str(value).zfill(2)}")
    month_summary = month_summary[["visit", "files"]]
    month_summary.to_csv(output_dir / "month_summary.csv", index=False)

    label_month = (
        df.groupby(["label", "month"])
        .size()
        .reset_index(name="files")
        .sort_values(["label", "month"])
    )
    label_month["visit"] = label_month["month"].apply(lambda value: f"m{str(value).zfill(2)}")
    label_month = label_month[["label", "visit", "files"]]
    label_month.to_csv(output_dir / "label_month_summary.csv", index=False)

    subject_months: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in df.itertuples(index=False):
        subject_months[(row.subject, row.label)].add(row.month)

    complete_rows = []
    required = {"00", "06", "12", "24"}
    labels = sorted(df["label"].unique())
    for label in labels:
        complete = sum(1 for (subject, lab), months in subject_months.items() if lab == label and required.issubset(months))
        complete_rows.append({"label": label, "complete_subjects_m00_m06_m12_m24": complete})
    pd.DataFrame(complete_rows).to_csv(output_dir / "complete_followup_summary.csv", index=False)


def summarize_headers(df: pd.DataFrame, output_dir: Path) -> None:
    shape_counter: Counter[tuple[int, ...]] = Counter()
    zoom_counter: Counter[tuple[float, ...]] = Counter()
    dtype_counter: Counter[str] = Counter()
    affine_counter: Counter[tuple[float, ...]] = Counter()

    for row in df.itertuples(index=False):
        img = nib.load(row.path)
        shape_counter[img.shape] += 1
        zoom_counter[tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])] += 1
        dtype_counter[str(img.get_data_dtype())] += 1
        affine_sig = tuple(round(float(x), 3) for x in img.affine[:3, :4].ravel())
        affine_counter[affine_sig] += 1

    pd.DataFrame(
        [{"shape": str(shape), "count": count} for shape, count in shape_counter.items()]
    ).to_csv(output_dir / "shape_summary.csv", index=False)
    pd.DataFrame(
        [{"zooms": str(zooms), "count": count} for zooms, count in zoom_counter.items()]
    ).to_csv(output_dir / "zoom_summary.csv", index=False)
    pd.DataFrame(
        [{"dtype": dtype, "count": count} for dtype, count in dtype_counter.items()]
    ).to_csv(output_dir / "dtype_summary.csv", index=False)
    pd.DataFrame(
        [{"affine_signature": str(signature), "count": count} for signature, count in affine_counter.items()]
    ).to_csv(output_dir / "affine_summary.csv", index=False)


def summarize_intensity(df: pd.DataFrame, output_dir: Path) -> None:
    baseline = df[df["month"] == "00"].copy()
    rows = []
    for label in sorted(baseline["label"].unique()):
        subset = baseline[baseline["label"] == label]
        zero_ratios = []
        means = []
        stds = []
        minima = []
        maxima = []
        nz_p95 = []
        nz_p99 = []
        for row in subset.itertuples(index=False):
            data = nib.load(row.path).get_fdata(dtype=np.float32)
            mask = data != 0
            zero_ratios.append(float((~mask).mean()))
            means.append(float(data.mean()))
            stds.append(float(data.std()))
            minima.append(float(data.min()))
            maxima.append(float(data.max()))
            if mask.any():
                nonzero = data[mask]
                nz_p95.append(float(np.percentile(nonzero, 95)))
                nz_p99.append(float(np.percentile(nonzero, 99)))

        rows.append(
            {
                "label": label,
                "n_samples_m00": int(len(subset)),
                "mean_zero_ratio": float(np.mean(zero_ratios)),
                "mean_intensity": float(np.mean(means)),
                "mean_std": float(np.mean(stds)),
                "min_of_min": float(np.min(minima)),
                "max_of_max": float(np.max(maxima)),
                "mean_nonzero_p95": float(np.mean(nz_p95)),
                "mean_nonzero_p99": float(np.mean(nz_p99)),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "baseline_intensity_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = index_dataset(data_dir)
    summarize_basic(df, output_dir)
    summarize_headers(df, output_dir)
    summarize_intensity(df, output_dir)
    print(f"saved profile tables to: {output_dir}")


if __name__ == "__main__":
    main()
