import cv2
import numpy as np

def detect_orb_features(image, visualize=True, max_features=500, scale_factor=1.2, nlevels=8):
    """
    Detect ORB features from an image and optionally visualize them.
    
    Args:
        image (numpy.ndarray): Input image (BGR format from ROS2 camera)
        visualize (bool): Whether to create a visualization image with keypoints and descriptors
        max_features (int): Maximum number of features to detect
        scale_factor (float): Pyramid decimation ratio
        nlevels (int): Number of pyramid levels
    
    Returns:
        tuple: (keypoints, descriptors, visualization_image)
            - keypoints: List of cv2.KeyPoint objects
            - descriptors: Numpy array of descriptors
            - visualization_image: Image with features drawn (None if visualize=False)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create ORB detector with customizable parameters
    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=scale_factor,
        nlevels=nlevels,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        patchSize=31,
        fastThreshold=20
    )
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Prepare return values
    vis_image = None
    
    # Create visualization if requested
    if visualize and keypoints:
        # Draw keypoints with rich information (size and orientation)
        vis_image = cv2.drawKeypoints(
            image, 
            keypoints, 
            None,  # Create a new image
            color=(0, 255, 0), 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Add text with keypoint count
        cv2.putText(
            vis_image,
            f"Features: {len(keypoints)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        # Optionally: visualize some of the binary descriptors if they exist
        if descriptors is not None and len(keypoints) > 0:
            # Select a subset of keypoints to display descriptors for 
            # (displaying all would make the image too cluttered)
            num_desc_to_show = min(5, len(keypoints))
            selected_indices = np.linspace(0, len(keypoints)-1, num_desc_to_show, dtype=int)
            
            # Draw sample descriptor patterns
            h, w = vis_image.shape[:2]
            descriptor_vis_area = np.ones((100, w, 3), dtype=np.uint8) * 255
            
            for i, idx in enumerate(selected_indices):
                if i >= num_desc_to_show:
                    break
                    
                # Get descriptor for this keypoint (binary descriptor)
                desc = descriptors[idx]
                
                # Calculate position to draw this descriptor
                start_x = int(w * (i + 0.5) / (num_desc_to_show + 1)) - 16
                
                # Draw a small 8x8 grid representation of part of the descriptor
                # (ORB descriptors are 256 bits, so we show just the first 64)
                for bit_idx in range(64):
                    row, col = bit_idx // 8, bit_idx % 8
                    bit_val = (desc[bit_idx // 8] >> (bit_idx % 8)) & 1
                    color = (0, 0, 0) if bit_val else (200, 200, 200)
                    x = start_x + col * 4
                    y = 20 + row * 4
                    cv2.rectangle(descriptor_vis_area, (x, y), (x+3, y+3), color, -1)
            
            # Combine the main image with descriptor visualization
            vis_image = np.vstack([vis_image, descriptor_vis_area])
    
    return keypoints, descriptors, vis_image

def process_ros2_image(ros2_image_msg):
    """
    Process ROS2 image message with ORB feature detection.
    This function should be called from your ROS2 node's subscription callback.
    
    Args:
        ros2_image_msg: ROS2 Image message
    """
    # Note: This assumes you've already converted the ROS2 image to OpenCV format
    # using cv_bridge or similar in your ROS2 node
    
    # Example integration with your ROS2 node:
    # from cv_bridge import CvBridge
    # bridge = CvBridge()
    # cv_image = bridge.imgmsg_to_cv2(ros2_image_msg, desired_encoding="bgr8")
    
    # Assuming cv_image is already available:
    cv_image = ros2_image_msg  # Replace with actual conversion in your node
    
    # Detect ORB features
    keypoints, descriptors, vis_image = detect_orb_features(cv_image)
    
    # Display the visualization
    if vis_image is not None:
        cv2.imshow("ORB Features", vis_image)
        cv2.waitKey(1)
    
    # Return the keypoints and descriptors for further processing if needed
    return keypoints, descriptors

if __name__ == "__main__":
    # Example usage with a test image
    test_image = cv2.imread("./ibvs_testing/ibvs_testing/handrail.jpg")
    keypoints, descriptors, vis_image = detect_orb_features(test_image)
    
    # Display the visualization
    if vis_image is not None:
        cv2.imshow("ORB Features", vis_image)
        cv2.waitKey(0)
    
    # Process the ROS2 image message (mock example)
    process_ros2_image(test_image)