import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_blobs(image_path):
     # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    

    
    # Set up the blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 0.01
    params.maxThreshold = 200
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 1  # Adjust based on expected blob size
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1  # Relaxed from perfect circle (1.0)
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.01
    
    # Filter by Inertia (measures how elongated a shape is)
    params.filterByInertia = True
    params.minInertiaRatio = 0.01  # Relaxed from perfect circle (1.0)
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(image)

     # Get image dimensions
    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)
    print(f"Image dimensions: {width}x{height}")
    print(f"Image center: {image_center}")

    if keypoints:
        print(f"Number of blobs detected: {len(keypoints)}")
        
        # Create a copy of the image to draw on
        output = image.copy()
        
        # Alternative approach for challenging cases: use binary thresholding
        # This section helps when SimpleBlobDetector struggles
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blob_centers = []
        
        # Process detected contours
        for i, contour in enumerate(contours):
            # Calculate centroid using moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Calculate area and approximate radius
                area = cv2.contourArea(contour)
                radius = int(np.sqrt(area / np.pi))
                
                # Only process if the blob is of reasonable size
                if area > 1000:  # Adjust threshold as needed
                    blob_centers.append((cX, cY, radius))
                    
                    # Draw the contour and center
                    cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(output, (cX, cY), 5, (0, 0, 255), -1)
                    
                    # Determine quadrant
                    quadrant = ""
                    if cX <= image_center[0] and cY <= image_center[1]:
                        quadrant = "Top-Left"
                    elif cX > image_center[0] and cY <= image_center[1]:
                        quadrant = "Top-Right"
                    elif cX <= image_center[0] and cY > image_center[1]:
                        quadrant = "Bottom-Left"
                    else:
                        quadrant = "Bottom-Right"
                    
                    # Calculate distance from center
                    dist_from_center = np.sqrt((cX - image_center[0])**2 + (cY - image_center[1])**2)
                    
                    # Print information about the blob
                    print(f"Blob {i+1} ({quadrant}):")
                    print(f"  Centroid: ({cX}, {cY})")
                    print(f"  Approximate radius: {radius} pixels")
                    print(f"  Area: {area} pixels²")
                    print(f"  Distance from image center: {dist_from_center:.2f} pixels")
                    print(f"  Relative position: {cX - image_center[0]}px horizontally, {cY - image_center[1]}px vertically")
                    print()

        output_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), 
                                                 (0, 0, 255), 
                                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Display both outputs
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title('Detected Blobs (Contour Method)')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(output_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Detected Blobs (SimpleBlobDetector)')
        plt.axis('off')
        
        plt.show()
    else:
        print("No blobs detected!")

def detect_circles(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=100,  # Minimum distance between circles
        param1=50,    # Upper threshold for Canny edge detector
        param2=30,    # Threshold for center detection
        minRadius=30, # Minimum radius to detect
        maxRadius=60  # Maximum radius to detect
    )
    
    # Get image dimensions
    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)
    print(f"Image dimensions: {width}x{height}")
    print(f"Image center: {image_center}")
    
    # Process and display circle information
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        print(f"Number of circles detected: {len(circles)}")
        
        # Create a copy of the image to draw on
        output = image.copy()
        
        # Analyze each circle
        for i, (x, y, r) in enumerate(circles):
            # Draw the circle and center
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
            
            # Calculate distance from center
            dist_from_center = np.sqrt((x - image_center[0])**2 + (y - image_center[1])**2)
            
            # Determine quadrant
            quadrant = ""
            if x <= image_center[0] and y <= image_center[1]:
                quadrant = "Top-Left"
            elif x > image_center[0] and y <= image_center[1]:
                quadrant = "Top-Right"
            elif x <= image_center[0] and y > image_center[1]:
                quadrant = "Bottom-Left"
            else:
                quadrant = "Bottom-Right"
                
            # Print information about the circle
            print(f"Circle {i+1} ({quadrant}):")
            print(f"  Center: ({x}, {y})")
            print(f"  Radius: {r} pixels")
            print(f"  Distance from image center: {dist_from_center:.2f} pixels")
            print(f"  Relative position: {x - image_center[0]:.2f}px horizontally, {y - image_center[1]:.2f}px vertically")
            print()
        
        # Calculate distances between circles
        if len(circles) > 1:
            print("Distances between circles:")
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, _ = circles[i]
                    x2, y2, _ = circles[j]
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    print(f"  Distance between circle {i+1} and circle {j+1}: {distance:.2f} pixels")
        
        # Display the output
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title('Detected Circles')
        plt.axis('off')
        plt.show()
    else:
        print("No circles detected!")

# Example usage (you would need to provide the correct path to your image)
# detect_circles('/home/tafarrel/ros2_ws/src/thesis_work/ibvs_testing/ibvs_testing/reference.jpg')
detect_blobs('/home/tafarrel/ros2_ws/src/thesis_work/ibvs_testing/ibvs_testing/BlobTest.jpg')