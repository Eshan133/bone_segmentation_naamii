# Knee Segmentation Project

This project processes knee CT scans to segment the femur and tibia, generate expanded and randomized masks, and identify the medial and lateral lowest points on the tibia surface for all masks. The code is organized into modular Python files for improved maintainability and reusability.

## Folder Structure

```
code/
├── setup.py
├── src/
│   ├── exception.py
│   ├── logger.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── io_utils.py
│   │   ├── segmentation.py
│   │   ├── analysis.py
│   │   ├── visualization.py
│   ├── main.py
├── data/
│   └── 3702_left_knee.nii.gz
├── output/
│   ├── viz_img/
│   │   ├── tibia_lowest_points_original.png
│   │   ├── tibia_points_all_masks.png
│   │   ├── segmentations_2mm_group.png
│   │   ├── segmentations_4mm_group.png
│   │   ├── knee_segmentation_original.png
│   │   ├── knee_segmentation_expanded_2mm.png
│   │   ├── knee_segmentation_expanded_4mm.png
│   │   ├── knee_segmentation_random_1.png
│   │   ├── knee_segmentation_random_2.png
│   ├── original_mask.nii.gz
│   ├── expanded_2mm_mask.nii.gz
│   ├── expanded_4mm_mask.nii.gz
│   ├── random_mask_1.nii.gz
│   ├── random_mask_2.nii.gz
│   ├── tibia_points_summary.csv
├── logs/
│   └── <timestamp>.log
├── tests/
├── README.md
├── requirements.txt
```

- `io_utils.py`: Handles loading and saving NIfTI files.
- `segmentation.py`: Contains segmentation-related functions.
- `analysis.py`: Computes tibia lowest points.
- `visualization.py`: Manages visualization of segmentations and points.
- `main.py`: Orchestrates the workflow.
- `data/input/`: Stores input CT scans.
- `data/output/`: Stores output masks and visualizations.
- `tests/`: Placeholder for test scripts.

## Installation

1. Clone the repository or create the folder structure.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the input CT scan (e.g., `3702_left_knee.nii.gz`) in `data/input/`.

## Usage

Run the main script from the `knee_segmentation` directory:
```bash
python main.py
```

This will:
- Load the CT scan.
- Segment the femur and tibia.
- Generate and save five masks: original, 2mm expanded, 4mm expanded, random_1, random_2.
- Calculate tibia lowest points for all masks.
- Generate visualizations saved in `data/output/`.
- Print a summary of volumes and tibia points.

## Dependencies

Listed in `requirements.txt`:
- nibabel
- numpy
- matplotlib
- scipy
- scikit-image

## Notes

- Ensure the input CT scan path in `main.py` matches your file location.
- If you have a custom folder structure, update paths in `main.py`.
- For testing, add scripts to the `tests/` directory using a framework like `pytest`.