import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_transformation(reference_image_path, test_image_path, visualize=True):
    """
    Find the transformation between reference and test images using SIFT.
    
    Args:
        reference_image_path: Path to the reference image
        test_image_path: Path to the test image
        visualize: Whether to visualize the matches and results
    
    Returns:
        transformation_matrix: The 3x3 homography matrix from reference to test
        status: True if successful, False otherwise
        inliers_ratio: Ratio of inliers to total matches
    """
    # Read images
    reference_img = cv2.imread(reference_image_path)
    test_img = cv2.imread(test_image_path)
    
    if reference_img is None or test_img is None:
        print("Error: Could not load image(s)")
        return None, False, 0
    
    # Convert to grayscale
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp_reference, des_reference = sift.detectAndCompute(reference_gray, None)
    kp_test, des_test = sift.detectAndCompute(test_gray, None)
    
    # show the reference descriptors using opencv
    reference_img = cv2.drawKeypoints(reference_gray ,
                      kp_reference ,
                      reference_img ,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Reference Image", reference_img)
    cv2.waitKey(1)

    test_img = cv2.drawKeypoints(test_gray ,
                      kp_test ,
                      test_img ,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Test Image", test_img)
    cv2.waitKey(1)

    if len(kp_reference) == 0 or len(kp_test) == 0:
        print("Error: No keypoints found in one or both images")
        return None, False, 0
    
    # FLANN parameters for fast matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors using KNN
    matches = flann.knnMatch(des_reference, des_test, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
    
    # Need at least 4 good matches to find homography
    if len(good_matches) < 4:
        print(f"Not enough good matches: {len(good_matches)}/4")
        return None, False, 0
    
    # Extract locations of matched keypoints
    reference_pts = np.float32([kp_reference[m.queryIdx].pt for m in good_matches])
    test_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches])
    
    # Find homography matrix using RANSAC
    H, mask = cv2.findHomography(reference_pts, test_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("Failed to find homography")
        return None, False, 0
    
    # Calculate inliers ratio
    inliers_count = np.sum(mask)
    inliers_ratio = inliers_count / len(good_matches)
    
    # Visualization
    if visualize:
        # Filter matches based on mask
        matched_mask = mask.ravel().tolist()
        
        # Draw matches
        result_img = cv2.drawMatches(reference_img, kp_reference, test_img, kp_test, 
                                     good_matches, None, 
                                     matchColor=(0, 255, 0),
                                     singlePointColor=None,
                                     matchesMask=matched_mask,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'SIFT Matches: {inliers_count}/{len(good_matches)} inliers')
        plt.tight_layout()
        plt.show()
        
        # Show transformation - warp reference to test image space
        h, w = reference_img.shape[:2]
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # Draw polygon around the transformed reference image
        result = test_img.copy()
        cv2.polylines(result, [np.int32(transformed_corners)], True, (0, 255, 0), 3)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Transformed reference image outline in test image')
        plt.tight_layout()
        plt.show()
    
    return H, True, inliers_ratio

def extract_transform_parameters(H):
    """
    Extract rotation, translation, and scale from homography matrix.
    This is a simplified version and works best for mostly planar scenes.
    
    Args:
        H: 3x3 homography matrix
    
    Returns:
        Dictionary containing rotation (degrees), translation (x,y), and scale
    """
    if H is None:
        return None
    
    # Decompose homography matrix
    # For a pure rotation+translation+scale in 2D, we can use:
    # a, b = H[0,0], H[0,1]
    # c, d = H[1,0], H[1,1]
    # tx, ty = H[0,2], H[1,2]
    
    a, b = H[0,0], H[0,1]
    c, d = H[1,0], H[1,1]
    tx, ty = H[0,2], H[1,2]
    
    # Calculate rotation angle in degrees
    rotation_rad = np.arctan2(b, a)
    rotation_deg = np.degrees(rotation_rad)
    
    # Calculate scale (this is a simplification - assumes isotropic scaling)
    scale = np.sqrt(a*a + b*b)
    
    return {
        'rotation': rotation_deg,
        'translation_x': tx,
        'translation_y': ty,
        'scale': scale
    }

# Example usage
if __name__ == "__main__":
    reference_path = "/home/tafarrel/ref.jpg"
    test_path = "/home/tafarrel/test2.jpg"
    
    H, status, inliers_ratio = find_transformation(reference_path, test_path)
    
    if status:
        print(f"Homography matrix:\n{H}")
        print(f"Inliers ratio: {inliers_ratio:.2f}")
        
        # Extract transformation parameters
        params = extract_transform_parameters(H)
        if params:
            print(f"Transformation parameters:")
            print(f"  Rotation: {params['rotation']:.2f} degrees")
            print(f"  Translation: ({params['translation_x']:.2f}, {params['translation_y']:.2f})")
            print(f"  Scale: {params['scale']:.2f}")
    else:
        print("Transformation estimation failed")