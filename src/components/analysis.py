import numpy as np

def find_tibia_lowest_points(segmentation_volume, voxel_sizes):
    """Find the medial and lateral lowest points on the tibia surface."""
    print("Finding lowest points on tibia surface...")
    
    tibia_mask = segmentation_volume == 1
    
    if np.sum(tibia_mask) == 0:
        print("No tibia detected in the segmentation.")
        return None, None, None, None
    
    tibia_indices = np.where(tibia_mask)
    
    tibia_points = np.array([tibia_indices[0] * voxel_sizes[0], 
                             tibia_indices[1] * voxel_sizes[1],
                             tibia_indices[2] * voxel_sizes[2]]).T
    
    center_x = np.mean(tibia_points[:, 0])
    
    lateral_mask = tibia_points[:, 0] < center_x
    medial_mask = tibia_points[:, 0] >= center_x
    
    lateral_points = tibia_points[lateral_mask]
    medial_points = tibia_points[medial_mask]
    
    if len(lateral_points) > 0:
        lateral_lowest_idx = np.argmin(lateral_points[:, 2])
        lateral_lowest = lateral_points[lateral_lowest_idx]
    else:
        lateral_lowest = None
        
    if len(medial_points) > 0:
        medial_lowest_idx = np.argmin(medial_points[:, 2])
        medial_lowest = medial_points[medial_lowest_idx]
    else:
        medial_lowest = None
    
    if lateral_lowest is not None:
        lateral_voxel = (
            int(lateral_lowest[0] / voxel_sizes[0]),
            int(lateral_lowest[1] / voxel_sizes[1]),
            int(lateral_lowest[2] / voxel_sizes[2])
        )
    else:
        lateral_voxel = None
        
    if medial_lowest is not None:
        medial_voxel = (
            int(medial_lowest[0] / voxel_sizes[0]),
            int(medial_lowest[1] / voxel_sizes[1]),
            int(medial_lowest[2] / voxel_sizes[2])
        )
    else:
        medial_voxel = None
    
    print(f"Tibia Medial lowest point (mm): {medial_lowest}")
    print(f"Tibia Lateral lowest point (mm): {lateral_lowest}")
    print(f"Tibia Medial lowest point (voxel): {medial_voxel}")
    print(f"Tibia Lateral lowest point (voxel): {lateral_voxel}")
    
    return medial_lowest, lateral_lowest, medial_voxel, lateral_voxel