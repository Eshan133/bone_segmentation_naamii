from skimage import morphology
from scipy import ndimage
import numpy as np

def create_bone_mask(slice_data, bone_threshold=200, min_size=50):
    """Create a binary mask for bones in a single slice."""
    bone_mask = slice_data > bone_threshold
    bone_mask = morphology.remove_small_objects(bone_mask, min_size=min_size)
    bone_mask = morphology.binary_closing(bone_mask, morphology.disk(2))
    return bone_mask

def find_joint_position(bone_mask, height):
    """Find the approximate joint position in a slice."""
    vertical_profile = np.sum(bone_mask, axis=0)
    smooth_profile = ndimage.gaussian_filter1d(vertical_profile, sigma=5)
    
    middle_start = height // 3
    middle_end = 2 * height // 3
    middle_profile = smooth_profile[middle_start:middle_end]
    
    if len(middle_profile) > 5:
        min_indices = [
            j for j in range(2, len(middle_profile) - 2)
            if middle_profile[j] < middle_profile[j-1] and middle_profile[j] < middle_profile[j+1]
        ]
        if min_indices:
            joint_idx = min_indices[np.argmin([middle_profile[j] for j in min_indices])]
            return middle_start + joint_idx
    return height // 2  # Fallback to middle point

def segment_slice(slice_data, bone_threshold=200):
    """Segment a single slice into femur and tibia."""
    height = slice_data.shape[1]
    bone_mask = create_bone_mask(slice_data, bone_threshold)
    
    if np.max(slice_data) < bone_threshold:
        return None, None
    
    joint_position = find_joint_position(bone_mask, height)
    
    femur_mask = bone_mask.copy()
    femur_mask[:, joint_position:] = False
    femur_mask = morphology.remove_small_objects(femur_mask, min_size=100)
    
    tibia_mask = bone_mask.copy()
    tibia_mask[:, :joint_position] = False
    tibia_mask = morphology.remove_small_objects(tibia_mask, min_size=100)
    
    return femur_mask, tibia_mask

def apply_3d_postprocessing(segmentation_volume):
    """Apply 3D post-processing to smooth and refine segmentation."""
    tibia_seg = segmentation_volume == 1
    femur_seg = segmentation_volume == 2
    
    struct_element = np.ones((3, 3, 3))
    tibia_seg = ndimage.binary_fill_holes(tibia_seg)
    femur_seg = ndimage.binary_fill_holes(femur_seg)
    tibia_seg = ndimage.binary_closing(tibia_seg, structure=struct_element)
    femur_seg = ndimage.binary_closing(femur_seg, structure=struct_element)
    tibia_seg = morphology.remove_small_objects(tibia_seg, min_size=1000)
    femur_seg = morphology.remove_small_objects(femur_seg, min_size=1000)
    
    result = np.zeros_like(segmentation_volume)
    result[tibia_seg] = 1
    result[femur_seg] = 2
    return result

def expand_segmentation(segmentation_volume, voxel_sizes, expansion_mm=2.0):
    """Expand tibia and femur masks by a specified distance in mm."""
    kernel_sizes = [int(np.ceil(expansion_mm / vs)) for vs in voxel_sizes]
    kernel_sizes = [max(1, ks) for ks in kernel_sizes]
    print(f"Expanding segmentation by {expansion_mm} mm (kernel sizes: {kernel_sizes})")
    
    struct_element = np.ones(tuple(kernel_sizes))
    tibia_seg = segmentation_volume == 1
    femur_seg = segmentation_volume == 2
    
    tibia_expanded = ndimage.binary_dilation(tibia_seg, structure=struct_element)
    femur_expanded = ndimage.binary_dilation(femur_seg, structure=struct_element)
    
    overlap = tibia_expanded & femur_expanded
    tibia_expanded[overlap] = tibia_seg[overlap]
    femur_expanded[overlap] = femur_seg[overlap]
    
    result = np.zeros_like(segmentation_volume)
    result[tibia_expanded] = 1
    result[femur_expanded] = 2
    return result

def generate_random_mask(original_segmentation, expanded_segmentation, random_value=0.5):
    """Generate a random mask between the original and expanded segmentations."""
    random_value = max(0.0, min(1.0, random_value))
    print(f"Generating random mask with randomization factor: {random_value}")
    
    result = np.zeros_like(original_segmentation)
    
    for label in [1, 2]:
        original_label_mask = original_segmentation == label
        expanded_label_mask = expanded_segmentation == label
        
        expansion_region = expanded_label_mask & ~original_label_mask
        
        expansion_size = np.sum(expansion_region)
        if expansion_size > 0:
            random_mask = np.random.random(expansion_size) < random_value
            temp_expansion = np.zeros_like(expansion_region, dtype=bool)
            temp_expansion[expansion_region] = random_mask
            result[original_label_mask | temp_expansion] = label
        else:
            result[original_label_mask] = label
    
    return result

def fallback_segmentation(data, start_slice, end_slice):
    """Apply fallback segmentation method for low bone volume cases."""
    segmentation_volume = np.zeros_like(data)
    
    for i in range(start_slice, end_slice):
        slice_data = data[:, i, :]
        bone_mask = slice_data > 150
        height = slice_data.shape[1]
        mid_point = height // 2
        
        femur_mask = bone_mask.copy()
        femur_mask[:, mid_point:] = False
        femur_mask = morphology.remove_small_objects(femur_mask, min_size=50)
        
        tibia_mask = bone_mask.copy()
        tibia_mask[:, :mid_point] = False
        tibia_mask = morphology.remove_small_objects(tibia_mask, min_size=50)
        
        segmentation_volume[:, i, :][tibia_mask] = 1
        segmentation_volume[:, i, :][femur_mask] = 2
    
    struct_element = np.ones((5, 5, 5))
    tibia_seg = ndimage.binary_closing(segmentation_volume == 1, structure=struct_element)
    femur_seg = ndimage.binary_closing(segmentation_volume == 2, structure=struct_element)
    tibia_seg = ndimage.binary_fill_holes(tibia_seg)
    femur_seg = ndimage.binary_fill_holes(femur_seg)
    tibia_seg = morphology.remove_small_objects(tibia_seg, min_size=1000)
    femur_seg = morphology.remove_small_objects(femur_seg, min_size=1000)
    
    result = np.zeros_like(data)
    result[tibia_seg] = 1
    result[femur_seg] = 2
    return result