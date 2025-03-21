# pnp_tracker.py
import numpy as np
import cv2


class PnPTracker:
    def __init__(self, cam_intrinsic_param=None, cam_distortion_param=None, corner_pts=None):
        """Initialize the PnP Tracker with camera parameters."""
        # Initialize state variables
        self.input_mode = False
        self.initialize_mode = False
        self.track_mode = False
        self.box_pts = []
        self.img_object = None

        self.record_mode = False
        self.record_num = 0
        self.t1, self.t2, self.t3 = [], [], []
        self.r1, self.r2, self.r3 = [], [], []

        # Initialize camera parameters or use defaults
        if cam_intrinsic_param is None:
            self.cam_intrinsic_param = np.array(
                [[514.04093664, 0.0, 320], [0.0, 514.87476583, 240], [0.0, 0.0, 1.0]]
            )
        else:
            self.cam_intrinsic_param = cam_intrinsic_param

        if cam_distortion_param is None:
            # self.cam_distortion_param = np.array(
            #     [2.68661165e-01, -1.31720458e00, -3.22098653e-03, -1.11578383e-03, 2.44470018e00]
            # )
            self.cam_distortion_param = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            self.cam_distortion_param = cam_distortion_param

        # Initialize ORB feature detector
        self.orb = cv2.ORB_create()

        # Variables to store the object reference
        self.pts2 = None
        self.right_bound = None
        self.left_bound = None
        self.lower_bound = None
        self.upper_bound = None

    # def mouse_callback(self, event, x, y, flags, param):
    #     """Callback function for mouse events to select object points."""
    #     if self.input_mode and event == cv2.EVENT_LBUTTONDOWN and len(self.box_pts) < 4:
    #         self.box_pts.append([x, y])
    #         self.current_frame = cv2.circle(
    #             self.current_frame, (x, y), 4, (0, 255, 0), 2
    #         )
    #         return True
    #     return False

    def start_object_selection(self, frame):
        """Start the object selection process."""
        self.current_frame = frame.copy()
        self.box_pts = []
        self.input_mode = True
        return self.current_frame

    def check_object_selection_complete(self):
        """Check if object selection is complete (4 points selected)."""
        if len(self.box_pts) >= 4:
            self.initialize_mode = True
            self.input_mode = False
            self._initialize_tracking()
            return True
        return False

    def _initialize_tracking(self):
        """Initialize tracking after object selection is complete."""
        # Set the boundary of reference object
        (
            self.pts2,
            self.right_bound,
            self.left_bound,
            self.lower_bound,
            self.upper_bound,
        ) = self._set_boundary_of_reference()

        # Do perspective transform to reference object
        self.img_object = self._input_perspective_transform()

        # Set track mode to true
        self.track_mode = True

    def _set_boundary_of_reference(self):
        """Set the boundary of the reference object."""
        box_pts = self.box_pts

        # Upper bound
        upper_bound = min(box_pts[0][1], box_pts[1][1])
        # Lower bound
        lower_bound = max(box_pts[2][1], box_pts[3][1])
        # Left bound
        left_bound = min(box_pts[0][0], box_pts[2][0])
        # Right bound
        right_bound = max(box_pts[1][0], box_pts[3][0])

        upper_left_point = [0, 0]
        upper_right_point = [(right_bound - left_bound), 0]
        lower_left_point = [0, (lower_bound - upper_bound)]
        lower_right_point = [(right_bound - left_bound), (lower_bound - upper_bound)]

        pts2 = np.float32(
            [upper_left_point, upper_right_point, lower_left_point, lower_right_point]
        )

        return pts2, right_bound, left_bound, lower_bound, upper_bound

    def _input_perspective_transform(self):
        """Perform perspective transform on the reference object."""
        pts1 = np.float32(self.box_pts)
        M = cv2.getPerspectiveTransform(pts1, self.pts2)
        img_object = cv2.warpPerspective(
            self.current_frame,
            M,
            (
                (self.right_bound - self.left_bound),
                (self.lower_bound - self.upper_bound),
            ),
        )
        return cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)

    def process_frame(self, frame):
        """Process a new frame for tracking."""
        self.current_frame = frame.copy()
        result_frame = frame.copy()

        if not self.track_mode:
            return result_frame

        # Feature detection and description
        kp1, des1, kp2, des2 = self._orb_feature_descriptor()

        # Check if features were detected
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return result_frame

        try:
            # Feature matching
            matches = self._brute_force_feature_matcher(kp1, des1, kp2, des2)

            if len(matches) < 4:  # Need at least 4 points for homography
                return result_frame

            # Find homography matrix
            M, mask = self._find_homography_object(kp1, kp2, matches)

            if M is None:  # Homography estimation failed
                return result_frame

            # Apply homography matrix using perspective transformation
            corner_camera_coord, center_camera_coord, object_points_3d, center_pts = (
                self._output_perspective_transform(M)
            )

            # Solve pnp using iterative LMA algorithm
            rotation, translation = self._iterative_solve_pnp(
                object_points_3d, corner_camera_coord
            )

            # Convert to centimeters
            translation = (40.0 / 53.0) * translation * 0.1

            # Convert to degrees
            rotation = rotation * 180.0 / np.pi

            # Draw box around object
            result_frame = self._draw_box_around_object(
                corner_camera_coord, result_frame, center_camera_coord
            )

            # Show object position and orientation value to frame
            result_frame = self._put_position_orientation_value_to_frame(
                translation, rotation, result_frame
            )

            # Record data if in record mode
            if self.record_mode:
                self._record_samples_data(translation, rotation)

            return result_frame, translation, rotation
        except Exception as e:
            print(f"Error in processing frame: {str(e)}")
            return result_frame, None, None

    def start_recording(self):
        """Start recording pose data."""
        self.record_mode = True
        self.record_num = 0
        self.t1, self.t2, self.t3 = [], [], []
        self.r1, self.r2, self.r3 = [], [], []

    def _orb_feature_descriptor(self):
        """Detect and compute ORB features."""
        kp1, des1 = self.orb.detectAndCompute(self.img_object, None)
        kp2, des2 = self.orb.detectAndCompute(self.current_frame, None)
        return kp1, des1, kp2, des2

    def _brute_force_feature_matcher(self, kp1, des1, kp2, des2):
        """Match features using Brute Force."""
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)

    def _find_homography_object(self, kp1, kp2, matches):
        """Find homography matrix between reference and image frame."""
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M, mask

    def _output_perspective_transform(self, M):
        """Apply homography matrix for perspective transformation."""
        h, w = self.img_object.shape
        corner_pts = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
        ).reshape(-1, 1, 2)
        center_pts = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
        corner_pts_3d = np.float32(
            [
                [-w / 2, -h / 2, 0],
                [-w / 2, (h - 1) / 2, 0],
                [(w - 1) / 2, (h - 1) / 2, 0],
                [(w - 1) / 2, -h / 2, 0],
            ]
        )
        corner_camera_coord = cv2.perspectiveTransform(corner_pts, M)
        center_camera_coord = cv2.perspectiveTransform(center_pts, M)
        return corner_camera_coord, center_camera_coord, corner_pts_3d, center_pts

    def _iterative_solve_pnp(self, object_points, image_points):
        """Solve PnP using iterative LMA algorithm."""
        image_points = image_points.reshape(-1, 2)
        retval, rotation, translation, _ = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.cam_intrinsic_param,
            self.cam_distortion_param,
            confidence=0.99,
        )
        return rotation, translation

    def _draw_box_around_object(self, dst, frame, center_camera_coord):
        """Draw box around the detected object."""
        center_camera_coord = center_camera_coord.astype(int)
        frame = cv2.circle(frame, (center_camera_coord[0][0][0], center_camera_coord[0][0][1]), 4, (0, 255, 0), 2)

        return cv2.polylines(frame, [np.int32(dst)], True, 255, 3)

    def _put_position_orientation_value_to_frame(self, translation, rotation, frame):
        """Display position and orientation values on the frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            frame, "position(cm)", (10, 30), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "x:" + str(round(translation[0][0], 2)),
            (250, 30),
            font,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "y:" + str(round(translation[1][0], 2)),
            (350, 30),
            font,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "z:" + str(round(translation[2][0], 2)),
            (450, 30),
            font,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "orientation(degree)",
            (10, 60),
            font,
            0.7,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "x:" + str(round(rotation[0][0], 2)),
            (250, 60),
            font,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "y:" + str(round(rotation[1][0], 2)),
            (350, 60),
            font,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "z:" + str(round(rotation[2][0], 2)),
            (450, 60),
            font,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return frame

    def _record_samples_data(self, translation, rotation):
        """Record sample data for statistics."""
        self.record_num += 1

        self.t1.append(translation[0][0])
        self.t2.append(translation[1][0])
        self.t3.append(translation[2][0])

        self.r1.append(rotation[0][0])
        self.r2.append(rotation[1][0])
        self.r3.append(rotation[2][0])

        # If 50 samples have been recorded, compute statistics
        if self.record_num >= 50:
            stats = self.get_recorded_data_stats()
            self.record_mode = False
            self.record_num = 0
            self.t1, self.t2, self.t3 = [], [], []
            self.r1, self.r2, self.r3 = [], [], []
            return stats
        return None

    def get_recorded_data_stats(self):
        """Compute and return statistics of recorded data."""
        # Convert to numpy arrays
        t1 = np.array(self.t1)
        t2 = np.array(self.t2)
        t3 = np.array(self.t3)
        r1 = np.array(self.r1)
        r2 = np.array(self.r2)
        r3 = np.array(self.r3)

        # Calculate statistics
        stats = {
            "translation": {
                "x": {"mean": np.mean(t1), "std": np.std(t1)},
                "y": {"mean": np.mean(t2), "std": np.std(t2)},
                "z": {"mean": np.mean(t3), "std": np.std(t3)},
            },
            "rotation": {
                "x": {"mean": np.mean(r1), "std": np.std(r1)},
                "y": {"mean": np.mean(r2), "std": np.std(r2)},
                "z": {"mean": np.mean(r3), "std": np.std(r3)},
            },
        }

        return stats

    def reset(self):
        """Reset the tracker state."""
        self.input_mode = False
        self.initialize_mode = False
        self.track_mode = False
        self.box_pts = []
        self.img_object = None
        self.record_mode = False
        self.record_num = 0
        self.t1, self.t2, self.t3 = [], [], []
        self.r1, self.r2, self.r3 = [], [], []
