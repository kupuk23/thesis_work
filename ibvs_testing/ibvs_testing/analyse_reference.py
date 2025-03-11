import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/home/tafarrel/ros2_ws/src/thesis_work/ibvs_testing/ibvs_testing/reference.jpg"

def detect_blobs(image_path, min_contour_area=1000):
     # Read the image
    image = cv2.imread(image_path, )

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 3)


    # apply morphology open then close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    # invert blob
    blob = (255 - blob)

    # display the image vs blob
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(thresh, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(blob, cmap='gray')
    plt.title('Blob Image')
    plt.axis('off')
    plt.show()

    
    
    # Set up the blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(image)

     # Get image dimensions
    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)
    print(f"Image dimensions: {width}x{height}")
    print(f"Image center: {image_center}")

    print(f"Number of blobs detected: {len(keypoints)}")
    
    

    # Get contours
    cnts = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Create a copy of the image to draw on
    output = image.copy()
    blob_centers = []

    # Process detected contours
    for i, contour in enumerate(cnts):
        # Calculate centroid using moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Calculate area and approximate radius
            area = cv2.contourArea(contour)
            radius = int(np.sqrt(area / np.pi))
            
            # Only process if the blob is of reasonable size
            if area > min_contour_area:  # Adjust threshold as needed
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

    # output_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), 
    #                                             (0, 0, 255), 
    #                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display both outputs
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(output, cmap='gray')
    plt.title('Detected Blobs (Contour Method)')
    plt.axis('off')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(output_with_keypoints, cv2.COLOR_BGR2RGB))
    # plt.title('Detected Blobs (SimpleBlobDetector)')
    # plt.axis('off')
    
    cv2.imshow("RESULT", output)
    cv2.waitKey(0)
    
# Example usage (you would need to provide the correct path to your image)
# detect_circles('/home/tafarrel/ros2_ws/src/thesis_work/ibvs_testing/ibvs_testing/reference.jpg')
detect_blobs(image_path)