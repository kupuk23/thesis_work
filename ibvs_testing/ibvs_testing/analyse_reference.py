import cv2
import numpy as np
import matplotlib.pyplot as plt

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
detect_circles('/home/tafarrel/ros2_ws/src/thesis_work/ibvs_testing/ibvs_testing/reference.jpg')