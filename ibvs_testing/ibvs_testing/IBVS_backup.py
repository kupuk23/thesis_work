from PIL import Image
import numpy as np
import cv2

# Load the image
image_path = '/home/tafarrel/ros2_ws/src/thesis_work/ibvs_testing/ibvs_testing/points.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to isolate the points (black points on a white background)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours and centroids of the blobs
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
centroids = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))

# Draw the centroids on the image
image_with_centroids = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for centroid in centroids:
    cv2.circle(image_with_centroids, centroid, 3, (0, 0, 255), -1)

# Display the image
# cv2.imshow("Image with centroids", image_with_centroids)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(centroids)

