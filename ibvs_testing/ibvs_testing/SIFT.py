import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def detect_circles_sift(img, template, min_match_count=2, visualization=True, debug=True):
    """
    Detect multiple circles in an image using SIFT features and return their centroids.
    Added detailed debug visualizations for each step.
    
    Parameters:
    -----------
    img : ndarray
        Input image where we want to detect circles
    template : ndarray
        Template image of a circle to match against
    min_match_count : int
        Minimum number of feature matches required to consider a detection
    visualization : bool
        If True, visualize the detection results
    debug : bool
        If True, show debug visualizations for each step
        
    Returns:
    --------
    centroids : list of tuples
        List of (x, y) coordinates for detected circle centroids
    """
    # Create copies for visualization
    display_img = img.copy()
    display_template = template.copy()
    
    # Convert images to grayscale if needed
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
        display_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        
    if len(template.shape) == 3:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray_template = template
        display_template = cv2.cvtColor(gray_template, cv2.COLOR_GRAY2BGR)
    
    if debug:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(display_template, cv2.COLOR_BGR2RGB))
        plt.title('Template Image')
        plt.tight_layout()
        plt.show()
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors with SIFT
    kp_template, des_template = sift.detectAndCompute(gray_template, None)
    kp_img, des_img = sift.detectAndCompute(gray_img, None)
    if debug:
        # Visualize keypoints
        img_with_kp = cv2.drawKeypoints(display_img, kp_img, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        template_with_kp = cv2.drawKeypoints(display_template, kp_template, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
        plt.title(f'Input Image Keypoints: {len(kp_img)}')
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(template_with_kp, cv2.COLOR_BGR2RGB))
        plt.title(f'Template Keypoints: {len(kp_template)}')
        plt.tight_layout()
        plt.show()
    
    # Check if enough keypoints were found
    if des_template is None or des_img is None or len(kp_template) < 2 or len(kp_img) < 2:
        print("ERROR: Not enough keypoints found in images")
        if des_template is None:
            print("Template descriptors are None")
        if des_img is None:
            print("Image descriptors are None")
        if len(kp_template) < 2:
            print(f"Only {len(kp_template)} keypoints found in template (need at least 2)")
        if len(kp_img) < 2:
            print(f"Only {len(kp_img)} keypoints found in image (need at least 2)")
        return []
    
    # FLANN parameters and matcher setup
    try:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des_template, des_img, k=2)
        
        print(f"Total matches found: {len(matches)}")
        
        # Fallback to brute force if FLANN fails or returns too few matches
        if len(matches) < 2:
            print("WARNING: FLANN matcher returned too few matches, trying Brute Force matcher")
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_template, des_img, k=2)
            print(f"Brute Force total matches: {len(matches)}")
    except Exception as e:
        print(f"ERROR with FLANN matcher: {e}")
        print("Falling back to Brute Force matcher")
        # Use Brute Force matcher as fallback
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_template, des_img, k=2)
        print(f"Brute Force total matches: {len(matches)}")
    
    if len(matches) == 0:
        print("ERROR: No matches found between template and image")
        return []
    
    # Store all the good matches as per Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"Number of good matches: {len(good_matches)}")
    
    if debug:
        # Draw matches
        matches_img = cv2.drawMatches(display_template, kp_template, 
                                      display_img, kp_img, 
                                      good_matches, None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Good Matches: {len(good_matches)}')
        plt.tight_layout()
        plt.show()
    
    if len(good_matches) < min_match_count:
        print(f"ERROR: Not enough good matches - {len(good_matches)}/{min_match_count}")
        print("Try adjusting min_match_count or improving the template image")
        return []
    
    # Extract the positions of matched keypoints
    matched_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches])
    
    if debug:
        # Visualize matched points on the image
        matched_img = display_img.copy()
        for pt in matched_pts:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(matched_img, (x, y), 5, (0, 255, 0), -1)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
        plt.title('Matched Points')
        plt.tight_layout()
        plt.show()
    
    # Use DBSCAN to cluster the keypoints (for multiple circle detection)
    # Adjust epsilon based on image size - this is crucial
    img_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    epsilon = 0.05 * img_diagonal  # 5% of the image diagonal
    min_samples = max(3, min(5, len(matched_pts) // 10))  # Dynamic adjustment
    
    print(f"DBSCAN parameters: epsilon={epsilon:.2f}, min_samples={min_samples}")
    
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(matched_pts)
    
    labels = db.labels_
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    print(f"DBSCAN found {n_clusters} clusters")
    print(f"Points per cluster: {[np.sum(labels == label) for label in unique_labels if label != -1]}")
    print(f"Noise points: {np.sum(labels == -1)}")
    
    if debug:
        # Visualize clusters
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        
        # Create colormap for clusters
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Black used for noise
                col = [0, 0, 0, 1]
            else:
                col = color
                
            mask = (labels == label)
            cluster_points = matched_pts[mask]
            
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        c=[col], marker='o', s=50)
            
            # Add cluster label
            if label != -1 and len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                plt.text(center[0], center[1], str(label), 
                         fontsize=12, weight='bold',
                         bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title('DBSCAN Clustering Results')
        plt.tight_layout()
        plt.show()
    
    # Calculate centroids for each cluster
    centroids = []
    clustered_points = {}
    
    for label in unique_labels:
        if label != -1:  # Skip noise points
            cluster = matched_pts[labels == label]
            if len(cluster) >= min_match_count:
                centroid_x = np.mean(cluster[:, 0])
                centroid_y = np.mean(cluster[:, 1])
                centroids.append((int(centroid_x), int(centroid_y)))
                clustered_points[label] = cluster
    
    print(f"Found {len(centroids)} centroids meeting the minimum point threshold")
    
    if visualization and centroids:
        # Final visualization
        plt.figure(figsize=(12, 10))
        
        # Draw the image
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        
        # Draw the centroids and estimated circles
        for i, centroid in enumerate(centroids):
            label = list(clustered_points.keys())[i]
            cluster = clustered_points[label]
            
            # Calculate radius as average distance from centroid to points
            distances = np.sqrt(np.sum((cluster - centroid)**2, axis=1))
            avg_radius = int(np.mean(distances))
            max_radius = int(np.max(distances))
            
            # Draw the centroid
            plt.plot(centroid[0], centroid[1], 'ro', markersize=10)
            plt.text(centroid[0] + 10, centroid[1] + 10, 
                     f"Cluster {label}\nPoints: {len(cluster)}", 
                     color='white', fontsize=12,
                     bbox=dict(facecolor='red', alpha=0.7))
            
            # Draw circle based on average radius
            circle = plt.Circle(centroid, avg_radius, color='r', fill=False, linewidth=2)
            plt.gca().add_artist(circle)
            
            # Draw larger circle based on max radius (dashed)
            circle_max = plt.Circle(centroid, max_radius, color='b', fill=False, 
                                   linewidth=1, linestyle='--')
            plt.gca().add_artist(circle_max)
        
        plt.title(f"Detected {len(centroids)} circles")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return centroids

def create_circle_template(size=100, thickness=2):
    """
    Create a template circle image
    """
    template = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - 5
    cv2.circle(template, center, radius, 255, thickness)
    return template

def test_with_synthetic_data(noise_level=10, circle_count=4, debug=True):
    """
    Test the circle detection algorithm with synthetic data
    """
    # Create a template circle
    template_size = 100
    template = create_circle_template(template_size)
    
    # Create a test image with multiple circles at different scales
    img_size = 800
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Ground truth for circles (center_x, center_y, radius)
    np.random.seed(42)  # For reproducibility
    circles_gt = []
    
    for _ in range(circle_count):
        # Generate random positions avoiding image edges
        padding = 150
        x = np.random.randint(padding, img_size - padding)
        y = np.random.randint(padding, img_size - padding)
        r = np.random.randint(40, 120)
        
        # Avoid overlapping circles
        if circles_gt:
            overlap = True
            attempts = 0
            while overlap and attempts < 20:
                overlap = False
                for (cx, cy, cr) in circles_gt:
                    # Check if circles are too close
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < (r + cr):
                        overlap = True
                        x = np.random.randint(padding, img_size - padding)
                        y = np.random.randint(padding, img_size - padding)
                        r = np.random.randint(40, 120)
                        break
                attempts += 1
            
            if attempts >= 20:
                continue  # Skip adding this circle if can't find non-overlapping position
        
        circles_gt.append((x, y, r))
        cv2.circle(img, (x, y), r, 255, 2)
    
    # Add some noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Add some random lines and shapes for more challenging detection
        for _ in range(3):
            pt1 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
            pt2 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
            cv2.line(img, pt1, pt2, 255, 1)
    
    # Detect circles
    centroids = detect_circles_sift(img, template, min_match_count=5, debug=debug)
    
    print("\nGround truth circles:")
    for i, (x, y, r) in enumerate(circles_gt):
        print(f"Circle {i+1}: center=({x}, {y}), radius={r}")
    
    print("\nDetected centroids:")
    for i, (x, y) in enumerate(centroids):
        print(f"Centroid {i+1}: ({x}, {y})")
        
    # Evaluate results
    if circles_gt and centroids:
        matches = []
        for gt_idx, (gt_x, gt_y, _) in enumerate(circles_gt):
            best_match = None
            min_dist = float('inf')
            for det_idx, (det_x, det_y) in enumerate(centroids):
                dist = np.sqrt((gt_x - det_x)**2 + (gt_y - det_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = (det_idx, dist)
            
            if best_match and best_match[1] < 50:  # 50px threshold for matching
                matches.append((gt_idx, best_match[0], best_match[1]))
        
        print("\nMatching results:")
        for gt_idx, det_idx, dist in matches:
            print(f"Ground truth circle {gt_idx+1} matched with detected centroid {det_idx+1} (distance: {dist:.2f}px)")
        
        precision = len(matches) / len(centroids) if centroids else 0
        recall = len(matches) / len(circles_gt) if circles_gt else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nPrecision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
    
    return img, template, centroids

def preprocess_for_circle_detection(img, debug=True):
    """
    Preprocess the image to enhance circle detection
    """
    # Make a copy
    processed = img.copy()
    
    # Convert to grayscale if needed
    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed
    
    if debug:
        plt.figure(figsize=(10, 5))
        plt.subplot(131)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if debug:
        plt.subplot(132)
        plt.imshow(blurred, cmap='gray')
        plt.title('Blurred')
    
    # Apply adaptive thresholding or Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    if debug:
        plt.subplot(133)
        plt.imshow(edges, cmap='gray')
        plt.title('Edges')
        plt.tight_layout()
        plt.show()
    
    return edges

def test_with_real_image(img_path, debug=True):
    """
    Test the circle detection algorithm with a real image
    
    Parameters:
    -----------
    img_path : str
        Path to the input image
    debug : bool
        If True, show debug visualizations
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"ERROR: Could not load image from {img_path}")
            return None, None, []
        
        # Preprocess image
        print("Preprocessing image...")
        preprocessed = preprocess_for_circle_detection(img, debug=debug)
        
        # Create a template
        print("Creating circle template...")
        template = create_circle_template(100)
        
        # Also try processing with Hough Circles for comparison
        if debug:
            print("Detecting circles using Hough transform for comparison...")
            circles = cv2.HoughCircles(
                preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                param1=100, param2=30, minRadius=10, maxRadius=100
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                img_with_circles = img.copy()
                
                for circle in circles:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    # Draw the outer circle
                    cv2.circle(img_with_circles, center, radius, (0, 255, 0), 2)
                    # Draw the center of the circle
                    cv2.circle(img_with_circles, center, 2, (0, 0, 255), 3)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
                plt.title(f'Hough Circles: {len(circles)} circles detected')
                plt.tight_layout()
                plt.show()
        
        # Detect circles using SIFT
        print("Detecting circles using SIFT...")
        centroids = detect_circles_sift(img, template, min_match_count=2, debug=debug)
        
        print(f"SIFT-based detection found {len(centroids)} circles")
        
        return img, template, centroids
    
    except Exception as e:
        print(f"ERROR during detection: {e}")
        import traceback
        traceback.print_exc()
        return None, None, []

# Run the test
if __name__ == "__main__":
    # Test with synthetic data by default
    print("=== Testing with synthetic data ===")
    # img, template, centroids = test_with_synthetic_data(noise_level=10, circle_count=4)
    
    # Uncomment to test with real image
    # print("\n=== Testing with real image ===")
    img, template, centroids = test_with_real_image("/home/tafarrel/ros2_ws/src/thesis_work/ibvs_testing/ibvs_testing/points.png")