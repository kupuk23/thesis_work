# Import necessary threading modules
import threading
import queue
import time
import cv2
import numpy as np

class PnPTrackerUI:
    """A class to handle OpenCV UI in a separate thread."""
    
    def __init__(self, pnp_tracker):
        """Initialize the UI handler with a reference to the PnP tracker."""
        self.tracker = pnp_tracker
        self.running = False
        self.selection_active = False
        self.frame_queue = queue.Queue(maxsize=1)  # Only need the latest frame
        self.ui_thread = None
        self.selection_complete = threading.Event()
        self.selection_started = threading.Event()
    
    def start(self):
        """Start the UI thread."""
        if self.ui_thread is not None and self.ui_thread.is_alive():
            return False  # Thread is already running
        
        self.running = True
        self.ui_thread = threading.Thread(target=self._ui_loop, daemon=True)
        self.ui_thread.start()
        return True
    
    def stop(self):
        """Stop the UI thread."""
        self.running = False
        if self.ui_thread is not None:
            self.ui_thread.join(timeout=1.0)
            self.ui_thread = None
    
    def update_frame(self, frame):
        """Update the frame to be displayed in the UI thread."""
        if not self.running:
            return
        
        # Replace any existing frame with the new one
        try:
            # Clear the queue of any old frames
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            # Put the new frame
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            # If the queue is full, try to clear it and add the new frame
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame.copy())
            except:
                pass
    
    def start_selection(self):
        """Start the object selection process."""
        if not self.running:
            return False
        
        self.selection_complete.clear()
        self.selection_active = True
        self.selection_started.set()
        return True
    
    def is_selection_complete(self):
        """Check if selection is complete."""
        return self.selection_complete.is_set()
    
    def wait_for_selection(self, timeout=None):
        """Wait for selection to complete and return success."""
        return self.selection_complete.wait(timeout)
    
    def _ui_loop(self):
        """Main UI thread loop."""
        # Create initial windows
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        
        last_frame = None
        
        while self.running:
            # Try to get the latest frame
            try:
                if not self.frame_queue.empty():
                    last_frame = self.frame_queue.get(timeout=0.1)
                
                # If we have a frame, show it
                if last_frame is not None:
                    cv2.imshow("Camera Feed", last_frame)
                
                # Handle selection mode
                if self.selection_active and self.selection_started.is_set():
                    self._handle_selection(last_frame)
                    self.selection_active = False
                    self.selection_started.clear()
                
                # Check for keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC key
                    self.running = False
                elif key == ord('i') and not self.selection_active:
                    # Trigger selection mode
                    self.start_selection()
                
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
                
            except Exception as e:
                print(f"Error in UI thread: {str(e)}")
                time.sleep(0.1)
        
        # Clean up
        cv2.destroyAllWindows()
    
    def _handle_selection(self, frame):
        """Handle the object selection process."""
        if frame is None:
            print("Cannot start selection: No frame available")
            self.selection_complete.set()
            return
        
        # Create a copy of the frame for selection
        selection_frame = frame.copy()
        
        # Initialize tracker with selection frame
        self.tracker.box_pts = []
        self.tracker.current_frame = selection_frame.copy()
        
        # Create selection window
        cv2.namedWindow("Object Selection", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Object Selection", self._selection_mouse_callback)
        
        # Wait for 4 points to be selected
        while len(self.tracker.box_pts) < 4 and self.running:
            cv2.imshow("Object Selection", self.tracker.current_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC key
                break
            time.sleep(0.05)
        
        # If 4 points were selected, initialize tracking
        if len(self.tracker.box_pts) == 4:
            # Initialize tracking mode
            self.tracker.initialize_mode = True
            self.tracker.input_mode = False
            self.tracker._initialize_tracking()
            print("Object selection complete. Tracking started.")
        else:
            print("Object selection cancelled or incomplete.")
        
        # Clean up
        cv2.destroyWindow("Object Selection")
        self.selection_complete.set()
    
    def _selection_mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for object selection."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.tracker.box_pts) < 4:
            self.tracker.box_pts.append([x, y])
            self.tracker.current_frame = cv2.circle(
                self.tracker.current_frame, (x, y), 4, (0, 255, 0), 2
            )
            print(f"Point {len(self.tracker.box_pts)} selected at ({x}, {y})")