# Knee Segmentation Project

This project processes knee CT scans to segment the femur and tibia, generate expanded and randomized masks, and identify the medial and lateral lowest points on the tibia surface for all masks. The package includes custom exception handling and logging for robust error tracking and debugging.

## Features

- **Segmentation**: Segments tibia (label 2, red) and femur (label 1, green) from knee CT scans using Otsu thresholding, watershed, and 3D post-processing.
- **Mask Variations**: Generates original, expanded (2mm and 4mm), and random (65% of 2mm, 40% of 4mm) segmentation masks.
- **Tibia Lowest Point Analysis**: Identifies medial and lateral lowest points on the tibia in voxel and mm coordinates for all masks.
- **Visualizations**: Produces coronal slice views, segmentation overlays, and tibia point plots, saved as PNG files.
- **CSV Summary**: Exports tibia lowest points (voxel and mm coordinates) to a CSV file.
- **Custom Exception Handling**: Uses `CustomException` to provide detailed error messages (file, line, message).
- **Logging**: Logs operations, warnings, and errors to timestamped files in `logs/` for debugging.

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
│   ├── **tibia_points_summary.csv**
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
- `data/`: Stores input CT scans.
- `output/`: Stores output masks, visualizations and coordinate.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/bone_segmentation.git
   cd bone_segmentation
   ```

2. **Create a Virtual Environment** (optional, but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This will also execute setup.py file


## Usage

1. **Prepare Input**:
   - Place your knee CT scan (NIfTI format) in `data/`, e.g., `data/3702_left_knee.nii.gz`.

2. **Run the Pipeline**:
   Use the console script to process the CT scan, generate masks, analyze tibia points, and save outputs:
   ```bash
   bone_segmentation
   ```
   - **Input**: `data/3702_left_knee.nii.gz`
   - **Outputs**:
     - Masks: `output/*.nii.gz`
     - Visualizations: `output/viz_img/*.png`
     - Summary: `output/tibia_points_summary.csv`
     - Logs: `logs/<timestamp>.log`

## Output Details

- **Segmentation Masks** (`output/`):
  - `original_mask.nii.gz`: Original femur and tibia segmentation.
  - `expanded_2mm_mask.nii.gz`: Expanded by 2mm.
  - `expanded_4mm_mask.nii.gz`: Expanded by 4mm.
  - `random_mask_1.nii.gz`: Random mask (65% of 2mm expansion).
  - `random_mask_2.nii.gz`: Random mask (40% of 4mm expansion).

- **Visualizations** (`output/viz_img/`):
  - `tibia_lowest_points_original.png`: Tibia points on original mask.
  - `tibia_points_all_masks.png`: Tibia points across all masks.
  - `segmentations_2mm_group.png`: Original, 2mm expanded, and random 1 masks.
  - `segmentations_4mm_group.png`: Original, 4mm expanded, and random 2 masks.
  - `knee_segmentation_*.png`: Individual segmentation overlays.

- **CSV Summary** (`output/tibia_points_summary.csv`):
  - Columns: `Mask`, `Medial_X_mm`, `Medial_Y_mm`, `Medial_Z_mm`, `Lateral_X_mm`, `Lateral_Y_mm`, `Lateral_Z_mm`, `Medial_X_voxel`, `Medial_Y_voxel`, `Medial_Z_voxel`, `Lateral_X_voxel`, `Lateral_Y_voxel`, `Lateral_Z_voxel`.
  - Rows for each mask (Original, Expanded 2mm, Expanded 4mm, Random 1, Random 2).

- **Logs** (`logs/<timestamp>.log`):
  - Detailed logs of operations, warnings (e.g., insufficient bone volume), and errors.
  - Example:
    ```
    [ 2025-05-22 22:48:00 ] 10 root - INFO - Starting bone_segmentation console script
    [ 2025-05-22 22:48:00 ] 15 root - INFO - Starting knee segmentation process
    ```

## Dependencies

Listed in `requirements.txt`:
- numpy
- nibabel
- matplotlib
- scipy
- scikit-image
- ipykernel
- pandas

## Notes

- Ensure the input CT scan path in `main.py` matches your file location.
- If you have a custom folder structure, update paths in `main.py`.
