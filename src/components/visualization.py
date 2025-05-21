import matplotlib.pyplot as plt
import numpy as np

def display_coronal_slice(data, title="Original Coronal Slice"):
    """Display the middle coronal slice of the volume."""
    coronal_idx = data.shape[1] // 2
    coronal_slice = data[:, coronal_idx, :].T
    
    plt.figure(figsize=(10, 10))
    plt.imshow(coronal_slice, cmap='gray', vmin=-300, vmax=1500)
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_segmentation(data, segmentation_volume, output_file='knee_segmentation.png', show_plot=True, 
                          title_suffix='', color_map='jet'):
    """Visualize segmentation overlay for given segmentation volume."""
    try:
        if show_plot:
            import matplotlib
            if matplotlib.get_backend().lower() in ['agg', 'cairo', 'ps', 'pdf']:
                print("Warning: Non-interactive backend detected. Switching to 'TkAgg' for display.")
                matplotlib.use('TkAgg')
        
        mid_slice = data.shape[1] // 2
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(data[:, mid_slice, :].T, cmap='gray', vmin=-300, vmax=1500)
        plt.title('Original Coronal Slice')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(segmentation_volume[:, mid_slice, :].T, cmap=color_map)
        plt.title(f'Segmentation Map {title_suffix}')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        overlay = np.zeros((*data[:, mid_slice, :].T.shape, 3))
        orig_slice = data[:, mid_slice, :].T
        orig_norm = np.clip(orig_slice, -300, 1500)
        orig_norm = (orig_norm - (-300)) / (1500 - (-300))
        
        overlay[:, :, 0] = orig_norm
        overlay[:, :, 1] = orig_norm
        overlay[:, :, 2] = orig_norm
        
        tibia_mask = segmentation_volume[:, mid_slice, :].T == 1
        femur_mask = segmentation_volume[:, mid_slice, :].T == 2
        
        overlay[femur_mask, 0] = 1.0
        overlay[femur_mask, 1] = 0.3
        overlay[femur_mask, 2] = 0.3
        overlay[tibia_mask, 0] = 0.3
        overlay[tibia_mask, 1] = 1.0
        overlay[tibia_mask, 2] = 0.3
        
        plt.imshow(overlay)
        plt.title(f'Segmentation Overlay {title_suffix}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Segmentation visualization saved to {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
                
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Visualization skipped. Check if matplotlib is properly installed and configured.")
        plt.close()

def visualize_all_segmentations(data, original_seg, expanded_seg, random_seg, 
                               output_file='all_segmentations.png', show_plot=True,
                               expansion_mm=2.0, random_value=0.5):
    """Visualize all three segmentation types side by side."""
    try:
        if show_plot:
            import matplotlib
            if matplotlib.get_backend().lower() in ['agg', 'cairo', 'ps', 'pdf']:
                print("Warning: Non-interactive backend detected. Switching to 'TkAgg' for display.")
                matplotlib.use('TkAgg')
        
        mid_slice = data.shape[1] // 2
        
        orig_slice = data[:, mid_slice, :].T
        orig_norm = np.clip(orig_slice, -300, 1500)
        orig_norm = (orig_norm - (-300)) / (1500 - (-300))
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot(2, 4, 1)
        plt.imshow(orig_norm, cmap='gray')
        plt.title('Original Coronal Slice')
        plt.axis('off')
        
        plt.subplot(2, 4, 2)
        plt.imshow(original_seg[:, mid_slice, :].T, cmap='jet')
        plt.title('Original Segmentation')
        plt.axis('off')
        
        plt.subplot(2, 4, 3)
        plt.imshow(expanded_seg[:, mid_slice, :].T, cmap='jet')
        plt.title(f'Expanded ({expansion_mm} mm)')
        plt.axis('off')
        
        plt.subplot(2, 4, 4)
        plt.imshow(random_seg[:, mid_slice, :].T, cmap='jet')
        plt.title(f'Random (factor: {random_value})')
        plt.axis('off')
        
        plt.subplot(2, 4, 6)
        orig_overlay = np.zeros((*orig_norm.shape, 3))
        orig_overlay[:, :, 0] = orig_norm
        orig_overlay[:, :, 1] = orig_norm
        orig_overlay[:, :, 2] = orig_norm
        
        tibia_mask = original_seg[:, mid_slice, :].T == 1
        femur_mask = original_seg[:, mid_slice, :].T == 2
        
        orig_overlay[femur_mask, 0] = 1.0
        orig_overlay[femur_mask, 1] = 0.3
        orig_overlay[femur_mask, 2] = 0.3
        orig_overlay[tibia_mask, 0] = 0.3
        orig_overlay[tibia_mask, 1] = 1.0
        orig_overlay[tibia_mask, 2] = 0.3
        
        plt.imshow(orig_overlay)
        plt.title('Original Overlay')
        plt.axis('off')
        
        plt.subplot(2, 4, 7)
        exp_overlay = np.zeros((*orig_norm.shape, 3))
        exp_overlay[:, :, 0] = orig_norm
        exp_overlay[:, :, 1] = orig_norm
        exp_overlay[:, :, 2] = orig_norm
        
        tibia_mask = expanded_seg[:, mid_slice, :].T == 1
        femur_mask = expanded_seg[:, mid_slice, :].T == 2
        
        exp_overlay[femur_mask, 0] = 1.0
        exp_overlay[femur_mask, 1] = 0.3
        exp_overlay[femur_mask, 2] = 0.3
        exp_overlay[tibia_mask, 0] = 0.3
        exp_overlay[tibia_mask, 1] = 1.0
        exp_overlay[tibia_mask, 2] = 0.3
        
        plt.imshow(exp_overlay)
        plt.title(f'Expanded Overlay ({expansion_mm} mm)')
        plt.axis('off')
        
        plt.subplot(2, 4, 8)
        rand_overlay = np.zeros((*orig_norm.shape, 3))
        rand_overlay[:, :, 0] = orig_norm
        rand_overlay[:, :, 1] = orig_norm
        rand_overlay[:, :, 2] = orig_norm
        
        tibia_mask = random_seg[:, mid_slice, :].T == 1
        femur_mask = random_seg[:, mid_slice, :].T == 2
        
        rand_overlay[femur_mask, 0] = 1.0
        rand_overlay[femur_mask, 1] = 0.3
        rand_overlay[femur_mask, 2] = 0.3
        rand_overlay[tibia_mask, 0] = 0.3
        rand_overlay[tibia_mask, 1] = 1.0
        rand_overlay[tibia_mask, 2] = 0.3
        
        plt.imshow(rand_overlay)
        plt.title(f'Random Overlay (factor: {random_value})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"All segmentations visualization saved to {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Visualization skipped. Check if matplotlib is properly installed and configured.")
        plt.close()

def visualize_tibia_points(data, segmentation_volume, tibia_medial_voxel, tibia_lateral_voxel, 
                          output_file='tibia_lowest_points.png', show_plot=True):
    """Visualize the medial and lateral lowest points on the tibia."""
    if tibia_medial_voxel is None or tibia_lateral_voxel is None:
        print("Cannot visualize tibia lowest points - missing coordinates.")
        return
    
    try:
        if show_plot:
            import matplotlib
            if matplotlib.get_backend().lower() in ['agg', 'cairo', 'ps', 'pdf']:
                print("Warning: Non-interactive backend detected. Switching to 'TkAgg' for display.")
                matplotlib.use('TkAgg')
        
        plt.figure(figsize=(15, 10))
        
        tm_x, tm_y, tm_z = tibia_medial_voxel
        tl_x, tl_y, tl_z = tibia_lateral_voxel
        
        avg_y = (tm_y + tl_y) // 2
        
        plt.subplot(2, 3, 1)
        plt.imshow(data[tm_x, :, :].T, cmap='gray', vmin=-300, vmax=1500)
        plt.scatter(tm_y, tm_z, c='g', marker='x', s=100, label='Tibia Medial')
        plt.title(f'Sagittal Slice (Tibia Medial: x={tm_x})')
        plt.legend()
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(data[:, avg_y, :].T, cmap='gray', vmin=-300, vmax=1500)
        plt.scatter(tm_x, tm_z, c='g', marker='x', s=100, label='Tibia Medial')
        plt.scatter(tl_x, tl_z, c='c', marker='x', s=100, label='Tibia Lateral')
        plt.title(f'Coronal Slice (y={avg_y})')
        plt.legend()
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(data[:, :, tm_z], cmap='gray', vmin=-300, vmax=1500)
        plt.scatter(tm_y, tm_x, c='g', marker='x', s=100, label='Tibia Medial')
        plt.title(f'Axial Slice (Tibia Medial: z={tm_z})')
        plt.legend()
        plt.axis('off')
        
        ax = plt.subplot(2, 3, 4, projection='3d')
        tibia_coords = np.where(segmentation_volume == 1)
        
        sample_rate = max(1, len(tibia_coords[0]) // 500)
        tx = tibia_coords[0][::sample_rate]
        ty = tibia_coords[1][::sample_rate]
        tz = tibia_coords[2][::sample_rate]
        
        ax.scatter(tx, ty, tz, c='lightgreen', alpha=0.1, marker='.', s=1, label='Tibia')
        ax.scatter([tm_x], [tm_y], [tm_z], c='g', marker='o', s=100, label='Tibia Medial')
        ax.scatter([tl_x], [tl_y], [tl_z], c='c', marker='o', s=100, label='Tibia Lateral')
        ax.set_title('3D Tibia with Lowest Points')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Tibia lowest points visualization saved to {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"Error during tibia points visualization: {e}")
        print("Visualization skipped.")
        plt.close()

def visualize_tibia_points_all_masks(data, masks, tibia_points, output_file='tibia_points_all_masks.png', show_plot=True):
    """Visualize tibia medial and lateral lowest points for all segmentation masks."""
    try:
        if show_plot:
            import matplotlib
            if matplotlib.get_backend().lower() in ['agg', 'cairo', 'ps', 'pdf']:
                print("Warning: Non-interactive backend detected. Switching to 'TkAgg' for display.")
                matplotlib.use('TkAgg')
        
        plt.figure(figsize=(20, 10))
        
        mask_names = ['Original', 'Expanded 2mm', 'Expanded 4mm', 'Random 1 (65%)', 'Random 2 (40%)']
        colors = ['g', 'c', 'm', 'y', 'b']
        
        mid_slice = data.shape[1] // 2
        orig_slice = data[:, mid_slice, :].T
        orig_norm = np.clip(orig_slice, -300, 1500)
        orig_norm = (orig_norm - (-300)) / (1500 - (-300))
        
        for i, (mask_name, mask, color) in enumerate(zip(mask_names, masks, colors)):
            medial_voxel = tibia_points[mask_name]['medial_voxel']
            lateral_voxel = tibia_points[mask_name]['lateral_voxel']
            
            if medial_voxel is None or lateral_voxel is None:
                print(f"Skipping visualization for {mask_name} - missing tibia points.")
                continue
            
            mx, my, mz = medial_voxel
            lx, ly, lz = lateral_voxel
            
            plt.subplot(2, 5, i + 1)
            plt.imshow(mask[:, mid_slice, :].T, cmap='jet')
            plt.scatter(mx, mz, c=color, marker='x', s=100, label=f'{mask_name} Medial')
            plt.scatter(lx, lz, c=color, marker='o', s=100, label=f'{mask_name} Lateral')
            plt.title(f'{mask_name} Tibia Points')
            plt.legend()
            plt.axis('off')
            
            plt.subplot(2, 5, i + 6)
            overlay = np.zeros((*orig_norm.shape, 3))
            overlay[:, :, 0] = orig_norm
            overlay[:, :, 1] = orig_norm
            overlay[:, :, 2] = orig_norm
            
            tibia_mask = mask[:, mid_slice, :].T == 1
            overlay[tibia_mask, 0] = 0.3
            overlay[tibia_mask, 1] = 1.0
            overlay[tibia_mask, 2] = 0.3
            
            plt.imshow(overlay)
            plt.scatter(mx, mz, c=color, marker='x', s=100, label=f'{mask_name} Medial')
            plt.scatter(lx, lz, c=color, marker='o', s=100, label=f'{mask_name} Lateral')
            plt.title(f'{mask_name} Overlay')
            plt.legend()
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Tibia points visualization for all masks saved to {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"Error during tibia points visualization: {e}")
        print("Visualization skipped.")
        plt.close()