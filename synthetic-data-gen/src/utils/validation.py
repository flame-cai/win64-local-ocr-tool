import numpy as np
from pathlib import Path

def validate_sample_files(sample_dir: Path) -> dict:
    """
    Loads a generated sample's files and checks for format errors.

    Args:
        sample_dir (Path): The directory of the sample to validate.

    Returns:
        dict: A dictionary containing validation results.
    """
    results = {"status": "OK", "errors": []}
    
    files_to_check = [
        "inputs_unnormalized.txt",
        "inputs_normalized.txt",
        "labels_textbox.txt",
        "labels_textline.txt",
    ]

    # Check file existence
    for f in files_to_check:
        if not (sample_dir / f).exists():
            results["status"] = "ERROR"
            results["errors"].append(f"Missing file: {f}")
    
    if results["status"] == "ERROR":
        return results

    # Load data and check lengths
    try:
        inputs_un = np.loadtxt(sample_dir / "inputs_unnormalized.txt")
        inputs_norm = np.loadtxt(sample_dir / "inputs_normalized.txt")
        labels_tb = np.loadtxt(sample_dir / "labels_textbox.txt", dtype=int)
        labels_tl = np.loadtxt(sample_dir / "labels_textline.txt", dtype=int)
    except Exception as e:
        results["status"] = "ERROR"
        results["errors"].append(f"Failed to load data files: {e}")
        return results

    num_points = inputs_un.shape[0]
    
    if not (num_points == inputs_norm.shape[0] == labels_tb.shape[0] == labels_tl.shape[0]):
        results["status"] = "ERROR"
        results["errors"].append(
            f"Mismatched line counts: "
            f"unnormalized={inputs_un.shape[0]}, "
            f"normalized={inputs_norm.shape[0]}, "
            f"textbox_labels={labels_tb.shape[0]}, "
            f"textline_labels={labels_tl.shape[0]}"
        )
    
    # Check data formats
    if inputs_un.ndim != 2 or inputs_un.shape[1] != 3:
        results["status"] = "ERROR"
        results["errors"].append("inputs_unnormalized.txt should have 3 columns.")
    
    if inputs_norm.ndim != 2 or inputs_norm.shape[1] != 3:
        results["status"] = "ERROR"
        results["errors"].append("inputs_normalized.txt should have 3 columns.")

    if labels_tb.ndim != 1:
        results["status"] = "ERROR"
        results["errors"].append("labels_textbox.txt should be a single column.")
    
    if labels_tl.ndim != 1:
        results["status"] = "ERROR"
        results["errors"].append("labels_textline.txt should be a single column.")

    # Check normalization range
    if np.any(inputs_norm[:, :2] < 0) or np.any(inputs_norm[:, :2] > 1):
        results["status"] = "WARNING"
        results["errors"].append("Normalized coordinates are outside [0, 1] range. This might be acceptable depending on jitter.")

    return results