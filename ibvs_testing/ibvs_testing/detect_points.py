import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations


def detect_circle_features(
    img,
    target_points,
    min_circle_radius=10,
    max_circle_radius=50,
    visualize=False,
    debug=False,
):
    """
    Detect 4 circle features using SIFT and return their centroids ordered from
    top-left to bottom-right.

    Args:
        img: Input image (BGR)
        min_circle_radius: Minimum radius of circle to detect
        max_circle_radius: Maximum radius of circle to detect
        visualize: Whether to display visualization of the detected circles

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

    # If visualization is enabled, draw the circles that were found in green
    if visualize:
        viz_img = img.copy()

        # visualize the blobbed image with circles instead of actual image
        viz_img = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
        for center in circle_centers:
            cv2.circle(viz_img, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
        cv2.putText(
            viz_img,
            f"Found {len(circle_centers)} circles",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    # We need exactly 4 circles
    if len(circle_centers) != 4:
        circle_centers = match_circles_opencv(circle_centers, target_points)

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

    for center in circle_centers:
        cv2.circle(viz_img, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)
    cv2.imshow("Circle Detection", viz_img)
    cv2.waitKey(1)

    return ordered_centers


def match_circles_opencv(detected_points, target_points, threshold=10.0):
    """
    Match detected circle centers to target pattern using OpenCV

    Parameters:
    detected_points: array of points detected in the current frame
    target_points: array of the 4 points in desired configuration

    Returns:
    matched_points: best 4 points matching the target pattern
    """
    if len(detected_points) < 4:
        return None

    # Convert to numpy arrays
    detected_points = np.array(detected_points, dtype=np.float32)
    target_points = np.array(target_points, dtype=np.float32)

    # Try all combinations of 4 detected points
    best_error = float("inf")
    best_points = None

    for indices in combinations(range(len(detected_points)), 4):
        src_points = detected_points[list(indices)]

        # Try all permutations of these 4 points
        for perm in permutations(range(4)):
            ordered_src = src_points[list(perm)]

            # Find homography
            try:
                H, _ = cv2.findHomography(ordered_src, target_points, cv2.RANSAC)

                if H is None:
                    continue

                # Transform points
                transformed = cv2.perspectiveTransform(
                    ordered_src.reshape(-1, 1, 2), H
                ).reshape(-1, 2)

                # Calculate error
                error = np.mean(np.linalg.norm(transformed - target_points, axis=1))

                if error < best_error:
                    best_error = error
                    best_points = ordered_src
            except:
                continue

    # Only return if error is below threshold
    if best_error > threshold:
        return None

    return best_points
