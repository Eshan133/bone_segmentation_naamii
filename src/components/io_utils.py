import nibabel as nib
import numpy as np

def load_nifti_image(input_path):
    """Load NIfTI image and return data, image object, and voxel sizes."""
    print(f"Loading image from {input_path}")
    img = nib.load(input_path)
    data = img.get_fdata()
    voxel_sizes = np.abs(img.affine.diagonal()[:3])  # Voxel sizes in mm (x, y, z)
    
    print(f"Image shape: {data.shape}")
    print(f"Image data type: {data.dtype}")
    print(f"Value range: [{np.min(data), np.max(data)}]")
    print(f"Voxel sizes (mm): {voxel_sizes}")
    
    return data, img, voxel_sizes

def save_segmentation(segmentation_volume, img, output_path):
    """Save segmentation volume as NIfTI file."""
    segmentation_nifti = nib.Nifti1Image(segmentation_volume.astype(np.int16), img.affine, img.header)
    segmentation_nifti.header.set_data_dtype(np.int16)
    nib.save(segmentation_nifti, output_path)
    print(f"Segmentation saved to {output_path}")