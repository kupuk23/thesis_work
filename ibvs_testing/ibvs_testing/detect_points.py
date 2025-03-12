import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_circle_features(
    img, min_circle_radius=10, max_circle_radius=50, visualize=False, debug=False
):
    """
    Detect 4 circle features using SIFT and return their centroids ordered from
    top-left to bottom-right.

    Args:
        img: Input image (BGR)
        min_circle_radius: Minimum radius of circle to detect
        max_circle_radius: Maximum radius of circle to detect
        debug: Whether to display visualization of the detected circles

    Returns:

        ordered_centers: Numpy array of shape (4, 2) containing the ordered
            centroids of the detected circles. If 4 circles are not found, returns None.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to handle varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 3
    )

    # apply morphology open then close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    # invert blob
    blob = 255 - blob

    # Find contours
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find circles
    circle_centers = []

    for contour in contours:
        # Calculate contour area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Skip very small contours
        if area < np.pi * (min_circle_radius**2):
            continue

        # Skip very large contours
        if area > np.pi * (max_circle_radius**2):
            continue

        # Calculate circularity (4π × area / perimeter²)
        # A perfect circle has circularity = 1
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)

        # Filter for circular shapes
        if circularity > 0.8:  # Threshold for "circle-like" shapes
            # Calculate moments to find centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                circle_centers.append((cx, cy))

    # We need exactly 4 circles
    if len(circle_centers) != 4 or debug:

        # If visualization is enabled, draw the circles that were found in green
        if visualize:
            viz_img = img.copy()

            # visualize the blobbed image with circles instead of actual image
            viz_img = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
            for center in circle_centers:
                cv2.circle(
                    viz_img, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1
                )
            cv2.putText(
                viz_img,
                f"Found {len(circle_centers)} circles",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Circle Detection", viz_img)
            cv2.waitKey(1)

        return None

    # Convert to numpy array
    centers = np.array(circle_centers, dtype=np.float32)

    # Order the points: top-left, top-right, bottom-left, bottom-right
    # First, sort by y-coordinate to separate top and bottom pairs
    centers = sorted(centers, key=lambda p: p[1])

    # Get top and bottom pairs
    top_pair = sorted(centers[:2], key=lambda p: p[0])  # Sort by x for top pair
    bottom_pair = sorted(centers[2:], key=lambda p: p[0])  # Sort by x for bottom pair

    # Combine into the final order: [top-left, top-right, bottom-left, bottom-right]
    ordered_centers = np.array(
        [top_pair[0], top_pair[1], bottom_pair[0], bottom_pair[1]], dtype=np.float32
    )

    return ordered_centers

    # # Apply SIFT to get more precise center if needed
    # roi = gray[
    #     max(0, cy - 20) : min(gray.shape[0], cy + 20),
    #     max(0, cx - 20) : min(gray.shape[1], cx + 20),
    # ]

    # if roi.size > 0:  # Make sure ROI is not empty
    #     keypoints = self.sift.detect(roi, None)

    #     # If SIFT finds keypoints in the ROI, refine the center
    #     if keypoints:
    #         strongest_kp = max(keypoints, key=lambda kp: kp.response)
    #         refined_x = strongest_kp.pt[0] + max(0, cx - 20)
    #         refined_y = strongest_kp.pt[1] + max(0, cy - 20)
    #         circle_centers.append((refined_x, refined_y))
    #     else:
    #         # If SIFT fails, use the centroid from moments
    #         circle_centers.append((cx, cy))
    # else:
    #     circle_centers.append((cx, cy))
