from __future__ import annotations

from pathlib import Path

from run_mri_svm_experiment import ExperimentConfig, run_experiment


def main() -> None:
    tasks = [
        ("ad_vs_normal_m00", ["AD", "NORMAL"], "00"),
        ("mci_vs_normal_m00", ["MCI", "NORMAL"], "00"),
        ("ad_vs_mci_m00", ["AD", "MCI"], "00"),
        ("multiclass_m00", ["AD", "MCI", "NORMAL"], "00"),
    ]

    for name, labels, month in tasks:
        print(f"\n=== Running {name} ===")
        config = ExperimentConfig(
            data_dir=Path("processed"),
            output_dir=Path("results") / name,
            cache_dir=Path("artifacts") / "feature_cache",
            labels=labels,
            month=month,
            target_shape=(32, 32, 32),
            pca_components=100,
            test_size=0.2,
            random_state=42,
            cv_folds=5,
            clip_percentile=99.5,
            n_jobs=-1,
        )
        run_experiment(config)


if __name__ == "__main__":
    main()
