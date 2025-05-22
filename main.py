import os
import numpy as np
import pandas as pd
import sys
from src.components.io_utils import load_nifti_image, save_segmentation
from src.components.segmentation import (segment_slice, apply_3d_postprocessing, 
                                        expand_segmentation, generate_random_mask, 
                                        fallback_segmentation)
from src.components.analysis import find_tibia_lowest_points
from src.components.visualization import (display_coronal_slice, visualize_segmentation, 
                                         visualize_all_segmentations, visualize_tibia_points, 
                                         visualize_tibia_points_all_masks)
from src.logger import logging
from src.exception import CustomException

def save_tibia_points_summary(tibia_points, output_file='output/tibia_points_summary.csv'):
    """Save tibia lowest points summary to a CSV file."""
    try:
        logging.info(f"Saving tibia points summary to {output_file}")
        data = []
        for mask_name, points in tibia_points.items():
            row = {'Mask': mask_name}
            medial_mm = points['medial_point_mm']
            lateral_mm = points['lateral_point_mm']
            row['Medial_X_mm'] = medial_mm[0] if medial_mm is not None else None
            row['Medial_Y_mm'] = medial_mm[1] if medial_mm is not None else None
            row['Medial_Z_mm'] = medial_mm[2] if medial_mm is not None else None
            row['Lateral_X_mm'] = lateral_mm[0] if lateral_mm is not None else None
            row['Lateral_Y_mm'] = lateral_mm[1] if lateral_mm is not None else None
            row['Lateral_Z_mm'] = lateral_mm[2] if lateral_mm is not None else None
            medial_voxel = points['medial_voxel']
            lateral_voxel = points['lateral_voxel']
            row['Medial_X_voxel'] = medial_voxel[0] if medial_voxel is not None else None
            row['Medial_Y_voxel'] = medial_voxel[1] if medial_voxel is not None else None
            row['Medial_Z_voxel'] = medial_voxel[2] if medial_voxel is not None else None
            row['Lateral_X_voxel'] = lateral_voxel[0] if lateral_voxel is not None else None
            row['Lateral_Y_voxel'] = lateral_voxel[1] if lateral_voxel is not None else None
            row['Lateral_Z_voxel'] = lateral_voxel[2] if lateral_voxel is not None else None
            data.append(row)
        
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully saved tibia points summary to {output_file}")
        print(f"Tibia points summary saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save tibia points summary to {output_file}")
        raise CustomException(e, sys)

def segment_knee(input_path, output_paths=None, show_plots=True):
    """Segment the femur and tibia, save masks, and find lowest points on tibia for all masks."""
    try:
        logging.info("Starting knee segmentation process")
        if not os.path.exists(input_path):
            logging.error(f"Input file does not exist: {input_path}")
            raise FileNotFoundError(f"Input file {input_path} does not exist.")
        
        if output_paths is None:
            output_dir = 'output'
            output_paths = {
                'original': os.path.join(output_dir, 'original_mask.nii.gz'),
                'expanded_2mm': os.path.join(output_dir, 'expanded_2mm_mask.nii.gz'),
                'expanded_4mm': os.path.join(output_dir, 'expanded_4mm_mask.nii.gz'),
                'random_1': os.path.join(output_dir, 'random_mask_1.nii.gz'),
                'random_2': os.path.join(output_dir, 'random_mask_2.nii.gz')
            }
        
        for path in output_paths.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logging.info(f"Ensured output directory exists for {path}")
        
        data, img, voxel_sizes = load_nifti_image(input_path)
        logging.info(f"Loaded input image: {input_path}, shape: {data.shape}, voxel sizes: {voxel_sizes}")
        if show_plots:
            logging.info("Displaying coronal slice")
            display_coronal_slice(data)
        
        segmentation_volume = np.zeros_like(data)
        start_slice = int(data.shape[1] * 0.2)
        end_slice = int(data.shape[1] * 0.8)
        
        logging.info(f"Segmenting slices from {start_slice} to {end_slice}")
        print("Segmenting knee CT...")
        for i in range(start_slice, end_slice):
            femur_mask, tibia_mask = segment_slice(data[:, i, :])
            if femur_mask is not None and tibia_mask is not None:
                segmentation_volume[:, i, :][tibia_mask] = 1
                segmentation_volume[:, i, :][femur_mask] = 2
        
        segmentation_volume = apply_3d_postprocessing(segmentation_volume)
        original_segmentation = segmentation_volume.copy()
        save_segmentation(original_segmentation, img, output_paths['original'])
        
        tibia_volume = np.sum(segmentation_volume == 1)
        femur_volume = np.sum(segmentation_volume == 2)
        logging.info(f"Original tibia volume: {tibia_volume} voxels, femur volume: {femur_volume} voxels")
        
        if tibia_volume < 1000 or femur_volume < 1000:
            logging.warning("Insufficient bone volume detected, applying fallback segmentation")
            print("Warning: Insufficient bone volume detected. Applying fallback method...")
            segmentation_volume = fallback_segmentation(data, start_slice, end_slice)
            original_segmentation = segmentation_volume.copy()
            save_segmentation(original_segmentation, img, output_paths['original'])
        
        expanded_2mm = expand_segmentation(original_segmentation, voxel_sizes, expansion_mm=2.0)
        save_segmentation(expanded_2mm, img, output_paths['expanded_2mm'])
        
        expanded_4mm = expand_segmentation(original_segmentation, voxel_sizes, expansion_mm=4.0)
        save_segmentation(expanded_4mm, img, output_paths['expanded_4mm'])
        
        random_1 = generate_random_mask(original_segmentation, expanded_2mm, random_value=0.65)
        save_segmentation(random_1, img, output_paths['random_1'])
        
        random_2 = generate_random_mask(original_segmentation, expanded_4mm, random_value=0.40)
        save_segmentation(random_2, img, output_paths['random_2'])
        
        masks = {
            'Original': original_segmentation,
            'Expanded 2mm': expanded_2mm,
            'Expanded 4mm': expanded_4mm,
            'Random 1 (65%)': random_1,
            'Random 2 (40%)': random_2
        }
        tibia_points = {}
        for mask_name, mask in masks.items():
            logging.info(f"Processing tibia points for {mask_name}")
            print(f"\nProcessing tibia points for {mask_name}...")
            medial_mm, lateral_mm, medial_voxel, lateral_voxel = find_tibia_lowest_points(mask, voxel_sizes)
            tibia_points[mask_name] = {
                'medial_point_mm': medial_mm,
                'lateral_point_mm': lateral_mm,
                'medial_voxel': medial_voxel,
                'lateral_voxel': lateral_voxel
            }
        
        output_dir = os.path.join('output', 'viz_img')
        logging.info("Visualizing tibia points for original segmentation")
        visualize_tibia_points(
            data, original_segmentation, 
            tibia_points['Original']['medial_voxel'], 
            tibia_points['Original']['lateral_voxel'],
            output_file=os.path.join(output_dir, 'tibia_lowest_points_original.png'), 
            show_plot=show_plots
        )
        
        logging.info("Visualizing tibia points for all masks")
        visualize_tibia_points_all_masks(
            data, 
            [original_segmentation, expanded_2mm, expanded_4mm, random_1, random_2],
            tibia_points,
            output_file=os.path.join(output_dir, 'tibia_points_all_masks.png'), 
            show_plot=show_plots
        )
        
        logging.info("Visualizing 2mm group segmentations")
        visualize_all_segmentations(
            data, original_segmentation, expanded_2mm, random_1, 
            output_file=os.path.join(output_dir, 'segmentations_2mm_group.png'), 
            show_plot=show_plots,
            expansion_mm=2.0, random_value=0.65
        )
        
        logging.info("Visualizing 4mm group segmentations")
        visualize_all_segmentations(
            data, original_segmentation, expanded_4mm, random_2, 
            output_file=os.path.join(output_dir, 'segmentations_4mm_group.png'), 
            show_plot=show_plots,
            expansion_mm=4.0, random_value=0.40
        )
        
        logging.info("Visualizing original segmentation")
        visualize_segmentation(
            data, original_segmentation, 
            output_file=os.path.join(output_dir, 'knee_segmentation_original.png'), 
            show_plot=show_plots,
            title_suffix='(Original)'
        )
        
        logging.info("Visualizing expanded 2mm segmentation")
        visualize_segmentation(
            data, expanded_2mm, 
            output_file=os.path.join(output_dir, 'knee_segmentation_expanded_2mm.png'), 
            show_plot=show_plots,
            title_suffix='(Expanded 2 mm)'
        )
        
        logging.info("Visualizing expanded 4mm segmentation")
        visualize_segmentation(
            data, expanded_4mm, 
            output_file=os.path.join(output_dir, 'knee_segmentation_expanded_4mm.png'), 
            show_plot=show_plots,
            title_suffix='(Expanded 4 mm)'
        )
        
        logging.info("Visualizing random 1 segmentation")
        visualize_segmentation(
            data, random_1, 
            output_file=os.path.join(output_dir, 'knee_segmentation_random_1.png'), 
            show_plot=show_plots,
            title_suffix='(Random 1: 65% of 2mm)'
        )
        
        logging.info("Visualizing random 2 segmentation")
        visualize_segmentation(
            data, random_2, 
            output_file=os.path.join(output_dir, 'knee_segmentation_random_2.png'), 
            show_plot=show_plots,
            title_suffix='(Random 2: 40% of 4mm)'
        )
        
        orig_tibia_volume = np.sum(original_segmentation == 1)
        orig_femur_volume = np.sum(original_segmentation == 2)
        logging.info(f"Original tibia volume: {orig_tibia_volume} voxels (label 1, green)")
        logging.info(f"Original femur volume: {orig_femur_volume} voxels (label 2, red)")
        print(f"Original tibia volume: {orig_tibia_volume} voxels (label 1, green)")
        print(f"Original femur volume: {orig_femur_volume} voxels (label 2, red)")
        
        exp_2mm_tibia = np.sum(expanded_2mm == 1)
        exp_2mm_femur = np.sum(expanded_2mm == 2)
        logging.info(f"2mm Expanded tibia volume: {exp_2mm_tibia} voxels")
        logging.info(f"2mm Expanded femur volume: {exp_2mm_femur} voxels")
        logging.info(f"2mm Expansion ratio: Tibia {exp_2mm_tibia/orig_tibia_volume:.2f}x, Femur {exp_2mm_femur/orig_femur_volume:.2f}x")
        print(f"2mm Expanded tibia volume: {exp_2mm_tibia} voxels")
        print(f"2mm Expanded femur volume: {exp_2mm_femur} voxels")
        print(f"2mm Expansion ratio: Tibia {exp_2mm_tibia/orig_tibia_volume:.2f}x, Femur {exp_2mm_femur/orig_femur_volume:.2f}x")
        
        exp_4mm_tibia = np.sum(expanded_4mm == 1)
        exp_4mm_femur = np.sum(expanded_4mm == 2)
        logging.info(f"4mm Expanded tibia volume: {exp_4mm_tibia} voxels")
        logging.info(f"4mm Expanded femur volume: {exp_4mm_femur} voxels")
        logging.info(f"4mm Expansion ratio: Tibia {exp_4mm_tibia/orig_tibia_volume:.2f}x, Femur {exp_4mm_femur/orig_femur_volume:.2f}x")
        print(f"4mm Expanded tibia volume: {exp_4mm_tibia} voxels")
        print(f"4mm Expanded femur volume: {exp_4mm_femur} voxels")
        print(f"4mm Expansion ratio: Tibia {exp_4mm_tibia/orig_tibia_volume:.2f}x, Femur {exp_4mm_femur/orig_femur_volume:.2f}x")
        
        rand_1_tibia = np.sum(random_1 == 1)
        rand_1_femur = np.sum(random_1 == 2)
        logging.info(f"Random 1 tibia volume: {rand_1_tibia} voxels")
        logging.info(f"Random 1 femur volume: {rand_1_femur} voxels")
        logging.info(f"Random 1 ratio: Tibia {rand_1_tibia/orig_tibia_volume:.2f}x, Femur {rand_1_femur/orig_femur_volume:.2f}x")
        print(f"Random 1 tibia volume: {rand_1_tibia} voxels")
        print(f"Random 1 femur volume: {rand_1_femur} voxels")
        print(f"Random 1 ratio: Tibia {rand_1_tibia/orig_tibia_volume:.2f}x, Femur {rand_1_femur/orig_femur_volume:.2f}x")
        
        rand_2_tibia = np.sum(random_2 == 1)
        rand_2_femur = np.sum(random_2 == 2)
        logging.info(f"Random 2 tibia volume: {rand_2_tibia} voxels")
        logging.info(f"Random 2 femur volume: {rand_2_femur} voxels")
        logging.info(f"Random 2 ratio: Tibia {rand_2_tibia/orig_tibia_volume:.2f}x, Femur {rand_2_femur/orig_femur_volume:.2f}x")
        print(f"Random 2 tibia volume: {rand_2_tibia} voxels")
        print(f"Random 2 femur volume: {rand_2_femur} voxels")
        print(f"Random 2 ratio: Tibia {rand_2_tibia/orig_tibia_volume:.2f}x, Femur {rand_2_femur/orig_femur_volume:.2f}x")
        
        logging.info("Generating tibia lowest points summary")
        for mask_name in masks:
            logging.info(f"Tibia points for {mask_name}:")
            logging.info(f"  Medial lowest point (mm): {tibia_points[mask_name]['medial_point_mm']}")
            logging.info(f"  Lateral lowest point (mm): {tibia_points[mask_name]['lateral_point_mm']}")
            logging.info(f"  Medial lowest point (voxel): {tibia_points[mask_name]['medial_voxel']}")
            logging.info(f"  Lateral lowest point (voxel): {tibia_points[mask_name]['lateral_voxel']}")
            print(f"\n{mask_name}:")
            print(f"  Medial lowest point (mm): {tibia_points[mask_name]['medial_point_mm']}")
            print(f"  Lateral lowest point (mm): {tibia_points[mask_name]['lateral_point_mm']}")
            print(f"  Medial lowest point (voxel): {tibia_points[mask_name]['medial_voxel']}")
            print(f"  Lateral lowest point (voxel): {tibia_points[mask_name]['lateral_voxel']}")
        
        save_tibia_points_summary(tibia_points, os.path.join('output', 'tibia_points_summary.csv'))
        
        logging.info("Completed knee segmentation process")
        return {
            'original': original_segmentation,
            'expanded_2mm': expanded_2mm,
            'expanded_4mm': expanded_4mm,
            'random_1': random_1,
            'random_2': random_2,
            'tibia_points': tibia_points
        }
    except Exception as e:
        logging.error("Error in knee segmentation process")
        raise CustomException(e, sys)

def run_main():
    """Entry point for console script."""
    try:
        logging.info("Starting bone_segmentation console script")
        input_path = os.path.join('data', '3702_left_knee.nii.gz')
        
        output_paths = {
            'original': os.path.join('output', 'original_mask.nii.gz'),
            'expanded_2mm': os.path.join('output', 'expanded_2mm_mask.nii.gz'),
            'expanded_4mm': os.path.join('output', 'expanded_4mm_mask.nii.gz'),
            'random_1': os.path.join('output', 'random_mask_1.nii.gz'),
            'random_2': os.path.join('output', 'random_mask_2.nii.gz')
        }
        
        results = segment_knee(input_path, output_paths, show_plots=True)
        
        logging.info("Final tibia lowest points summary:")
        for mask_name in results['tibia_points']:
            logging.info(f"{mask_name}:")
            logging.info(f"  Medial lowest point (mm): {results['tibia_points'][mask_name]['medial_point_mm']}")
            logging.info(f"  Lateral lowest point (mm): {results['tibia_points'][mask_name]['lateral_point_mm']}")
            logging.info(f"  Medial lowest point (voxel): {results['tibia_points'][mask_name]['medial_voxel']}")
            logging.info(f"  Lateral lowest point (voxel): {results['tibia_points'][mask_name]['lateral_voxel']}")
            print(f"\n{mask_name}:")
            print(f"  Medial lowest point (mm): {results['tibia_points'][mask_name]['medial_point_mm']}")
            print(f"  Lateral lowest point (mm): {results['tibia_points'][mask_name]['lateral_point_mm']}")
            print(f"  Medial lowest point (voxel): {results['tibia_points'][mask_name]['medial_voxel']}")
            print(f"  Lateral lowest point (voxel): {results['tibia_points'][mask_name]['lateral_voxel']}")
        
        logging.info("Console script completed successfully")
    except Exception as e:
        logging.error("Error in console script execution")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_main()