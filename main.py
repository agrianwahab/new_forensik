#!/usr/bin/env python3
"""
Advanced Forensic Image Analysis System v2.0
Main execution file

Usage:
    python main.py <image_path> [options]

Example:
    python main.py test_image.jpg
    python main.py test_image.jpg --export-all
    python main.py test_image.jpg --output-dir ./results
"""

import sys
import os
import time
import argparse
from PIL import Image#!/usr/bin/env python3
"""
Advanced Forensic Image Analysis System v2.0
Main execution file

Usage:
    python main.py <image_path> [options]

Example:
    python main.py test_image.jpg
    python main.py test_image.jpg --export-all
    python main.py test_image.jpg --output-dir ./results
"""

import sys
import os
import time
import argparse
import numpy as np    # PERBAIKAN: Tambah import numpy
import cv2           # PERBAIKAN: Tambah import cv2
from PIL import Image

# Import semua modul
from validation import validate_image_file, extract_enhanced_metadata, advanced_preprocess_image
from ela_analysis import perform_multi_quality_ela
from feature_detection import extract_multi_detector_features
from copy_move_detection import detect_copy_move_advanced, detect_copy_move_blocks, kmeans_tampering_localization
from advanced_analysis import (analyze_noise_consistency, analyze_frequency_domain, 
                              analyze_texture_consistency, analyze_edge_consistency,
                              analyze_illumination_consistency, perform_statistical_analysis)
from jpeg_analysis import advanced_jpeg_analysis, jpeg_ghost_analysis
from classification import classify_manipulation_advanced, prepare_feature_vector
from visualization import visualize_results_advanced, export_kmeans_visualization
from export_utils import export_complete_package


def analyze_image_comprehensive_advanced(image_path, output_dir="./results"):
    """Advanced comprehensive image analysis pipeline"""
    print(f"\n{'='*80}")
    print(f"ADVANCED FORENSIC IMAGE ANALYSIS SYSTEM v2.0")
    print(f"Enhanced Detection: Copy-Move, Splicing, Authentic Images")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Validation
    try:
        validate_image_file(image_path)
        print("✅ [1/17] File validation passed")
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return None
    
    # 2. Load image
    try:
        original_image = Image.open(image_path)
        print(f"✅ [2/17] Image loaded: {os.path.basename(image_path)}")
        print(f"  Size: {original_image.size}, Mode: {original_image.mode}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # 3. Enhanced metadata extraction
    print("🔍 [3/17] Extracting enhanced metadata...")
    metadata = extract_enhanced_metadata(image_path)
    print(f"  Authenticity Score: {metadata['Metadata_Authenticity_Score']}/100")
    
    # 4. Advanced preprocessing
    print("🔧 [4/17] Advanced preprocessing...")
    preprocessed, original_preprocessed = advanced_preprocess_image(original_image.copy())
    
    # 5. Multi-quality ELA
    print("📊 [5/17] Multi-quality Error Level Analysis...")
    ela_image, ela_mean, ela_std, ela_regional, ela_quality_stats, ela_variance = perform_multi_quality_ela(preprocessed.copy())
    print(f"  ELA Stats: μ={ela_mean:.2f}, σ={ela_std:.2f}, Regions={ela_regional['outlier_regions']}")
    
    # 6. Multi-detector feature extraction
    print("🎯 [6/17] Multi-detector feature extraction...")
    feature_sets, roi_mask, gray_enhanced = extract_multi_detector_features(
        preprocessed.copy(), ela_image, ela_mean, ela_std)
    total_features = sum(len(kp) for kp, _ in feature_sets.values())
    print(f"  Total keypoints: {total_features}")
    
    # 7. Advanced copy-move detection
    print("🔄 [7/17] Advanced copy-move detection...")
    ransac_matches, ransac_inliers, transform = detect_copy_move_advanced(
        feature_sets, preprocessed.size)
    print(f"  RANSAC inliers: {ransac_inliers}")
    
    # 8. Enhanced block matching
    print("🧩 [8/17] Enhanced block-based detection...")
    block_matches = detect_copy_move_blocks(preprocessed)
    print(f"  Block matches: {len(block_matches)}")
    
    # 9. Advanced noise analysis
    print("📡 [9/17] Advanced noise consistency analysis...")
    noise_analysis = analyze_noise_consistency(preprocessed)
    print(f"  Noise inconsistency: {noise_analysis['overall_inconsistency']:.3f}")
    
    # 10. Advanced JPEG analysis
    print("📷 [10/17] Advanced JPEG artifact analysis...")
    try:
        from jpeg_analysis import advanced_jpeg_analysis, jpeg_ghost_analysis
        jpeg_analysis = advanced_jpeg_analysis(preprocessed)
        
        # Robust handling untuk return values dari jpeg_ghost_analysis
        jpeg_ghost_result = jpeg_ghost_analysis(preprocessed)
        
        if len(jpeg_ghost_result) == 2:
            ghost_map, ghost_suspicious = jpeg_ghost_result
            ghost_analysis_details = {}
        elif len(jpeg_ghost_result) == 3:
            ghost_map, ghost_suspicious, ghost_analysis_details = jpeg_ghost_result
        else:
            raise ValueError(f"Unexpected return values from jpeg_ghost_analysis: {len(jpeg_ghost_result)}")
        
        ghost_ratio = np.sum(ghost_suspicious) / ghost_suspicious.size
        print(f"  JPEG anomalies: {ghost_ratio:.1%}")
        
    except Exception as e:
        print(f"  ⚠ JPEG analysis failed: {e}")
        # Fallback values
        jpeg_analysis = {
            'quality_responses': [],
            'response_variance': 0.0,
            'double_compression_indicator': 0.0,
            'estimated_original_quality': 0,
            'compression_inconsistency': False
        }
        ghost_map = np.zeros((preprocessed.size[1], preprocessed.size[0]))
        ghost_suspicious = np.zeros((preprocessed.size[1], preprocessed.size[0]), dtype=bool)
        ghost_analysis_details = {}
        ghost_ratio = 0.0

    
    # 11. Frequency domain analysis
    print("🌊 [11/17] Frequency domain analysis...")
    frequency_analysis = analyze_frequency_domain(preprocessed)
    print(f"  Frequency inconsistency: {frequency_analysis['frequency_inconsistency']:.3f}")
    
    # 12. Texture consistency analysis
    print("🧵 [12/17] Texture consistency analysis...")
    texture_analysis = analyze_texture_consistency(preprocessed)
    print(f"  Texture inconsistency: {texture_analysis['overall_inconsistency']:.3f}")
    
    # 13. Edge consistency analysis
    print("📐 [13/17] Edge density analysis...")
    edge_analysis = analyze_edge_consistency(preprocessed)
    print(f"  Edge inconsistency: {edge_analysis['edge_inconsistency']:.3f}")
    
    # 14. Illumination analysis
    print("💡 [14/17] Illumination consistency analysis...")
    illumination_analysis = analyze_illumination_consistency(preprocessed)
    print(f"  Illumination inconsistency: {illumination_analysis['overall_illumination_inconsistency']:.3f}")
    
    # 15. Statistical analysis
    print("📈 [15/17] Statistical analysis...")
    statistical_analysis = perform_statistical_analysis(preprocessed)
    print(f"  Overall entropy: {statistical_analysis['overall_entropy']:.3f}")
    
    # Prepare comprehensive results
    analysis_results = {
        'metadata': metadata,
        'ela_image': ela_image,
        'ela_mean': ela_mean,
        'ela_std': ela_std,
        'ela_regional_stats': ela_regional,
        'ela_quality_stats': ela_quality_stats,
        'ela_variance': ela_variance,
        'feature_sets': feature_sets,
        'sift_keypoints': feature_sets['sift'][0],
        'sift_descriptors': feature_sets['sift'][1],
        'sift_matches': len(ransac_matches),
        'ransac_matches': ransac_matches,
        'ransac_inliers': ransac_inliers,
        'geometric_transform': transform,
        'block_matches': block_matches,
        'noise_analysis': noise_analysis,
        'noise_map': cv2.cvtColor(np.array(preprocessed), cv2.COLOR_RGB2GRAY),
        'jpeg_analysis': jpeg_analysis,
        'jpeg_ghost': ghost_map,
        'jpeg_ghost_suspicious_ratio': ghost_ratio,
        'frequency_analysis': frequency_analysis,
        'texture_analysis': texture_analysis,
        'edge_analysis': edge_analysis,
        'illumination_analysis': illumination_analysis,
        'statistical_analysis': statistical_analysis,
        'color_analysis': {'illumination_inconsistency': illumination_analysis['overall_illumination_inconsistency']},
        'roi_mask': roi_mask,
        'enhanced_gray': gray_enhanced
    }
    
    # 16. Advanced tampering localization
    print("🎯 [16/17] Advanced tampering localization...")
    localization_results = advanced_tampering_localization(preprocessed, analysis_results)
    print(f"  Tampering area: {localization_results['tampering_percentage']:.1f}% of image")
    
    # 17. Advanced classification
    print("🤖 [17/17] Advanced manipulation classification...")
    classification = classify_manipulation_advanced(analysis_results)
    analysis_results['classification'] = classification
    analysis_results['localization_analysis'] = localization_results
    
    processing_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE - Processing Time: {processing_time:.2f}s")
    print(f"{'='*80}")
    print(f"📊 FINAL RESULT: {classification['type']}")
    print(f"📊 CONFIDENCE: {classification['confidence']}")
    print(f"📊 Copy-Move Score: {classification['copy_move_score']}/100")
    print(f"📊 Splicing Score: {classification['splicing_score']}/100")
    print(f"📊 Processing Time: {processing_time:.2f}s")
    print(f"{'='*80}\n")
    
    if classification['details']:
        print("📋 Detection Details:")
        for detail in classification['details']:
            print(f"  {detail}")
        print()
    
    return analysis_results

def advanced_tampering_localization(image_pil, analysis_results):
    """Advanced tampering localization menggunakan multiple methods"""
    print("🎯 Advanced tampering localization...")
    
    ela_image = analysis_results['ela_image']
    
    # 1. K-means based localization
    kmeans_result = kmeans_tampering_localization(image_pil, ela_image)
    
    # 2. Threshold-based localization
    ela_array = np.array(ela_image)
    threshold = analysis_results['ela_mean'] + 2 * analysis_results['ela_std']
    threshold_mask = ela_array > threshold
    
    # 3. Combined localization
    combined_mask = np.logical_or(
        kmeans_result['tampering_mask'],
        threshold_mask
    )
    
    # Morphological operations untuk clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    h, w = ela_array.shape
    
    return {
        'kmeans_localization': kmeans_result,
        'threshold_mask': threshold_mask,
        'combined_tampering_mask': combined_mask.astype(bool),
        'tampering_percentage': np.sum(combined_mask) / (h * w) * 100
    }

def main():
    parser = argparse.ArgumentParser(description='Advanced Forensic Image Analysis System v2.0')
    parser.add_argument('image_path', help='Path to the image file to analyze')
    parser.add_argument('--output-dir', '-o', default='./results', 
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--export-all', '-e', action='store_true',
                       help='Export complete package (PNG, PDF, DOCX, etc.)')
    parser.add_argument('--export-vis', '-v', action='store_true',
                       help='Export only visualization')
    parser.add_argument('--export-report', '-r', action='store_true',
                       help='Export only DOCX report')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file '{args.image_path}' not found!")
        sys.exit(1)
    
    # Run analysis
    try:
        analysis_results = analyze_image_comprehensive_advanced(args.image_path, args.output_dir)
        
        if analysis_results is None:
            print("❌ Analysis failed!")
            sys.exit(1)
        
        # Load original image for export
        original_image = Image.open(args.image_path)
        
        # Create base filename
        base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
        base_path = os.path.join(args.output_dir, base_filename)
        
        # Export based on arguments
        if args.export_all:
            print("\n📦 Exporting complete package...")
            export_complete_package(original_image, analysis_results, base_path)
        elif args.export_vis:
            print("\n📊 Exporting visualization...")
            visualize_results_advanced(original_image, analysis_results, f"{base_path}_analysis.png")
        elif args.export_report:
            print("\n📄 Exporting DOCX report...")
            from export_utils import export_to_advanced_docx
            export_to_advanced_docx(original_image, analysis_results, f"{base_path}_report.docx")
        else:
            # Default: export visualization
            print("\n📊 Exporting basic visualization...")
            visualize_results_advanced(original_image, analysis_results, f"{base_path}_analysis.png")
        
        print("✅ Analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
