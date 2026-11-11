import cv2
import numpy as np
import os

class DepthEstimator:
    """
    Monocular Depth Estimation using Intel's MiDaS v2.1 Small (ONNX).
    Fuses absolute disparity scaling with geometric perspective analysis
    to prevent false alarms in enclosed rooms and on distant background objects.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "model-small.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MiDaS model not found at {model_path}")

        self.net = cv2.dnn.readNet(model_path)
        self.input_size = (256, 256)
        # ImageNet normalization mean & scale
        self.mean = (123.675, 116.28, 103.53)
        self.scale = 1.0 / 255.0

    def estimate_depth(self, frame):
        """
        Runs MiDaS on frame and returns:
          depth_map: 2D float array (raw disparity) resized to frame resolution.
                     Higher values mean closer objects.
          depth_colored: 3-channel BGR colormap for visualization.
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, self.scale, self.input_size, self.mean, swapRB=True, crop=False
        )
        self.net.setInput(blob)
        output = self.net.forward()

        # MiDaS small raw disparity output
        depth_raw = output[0]
        depth_map = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_CUBIC)

        # Normalize 0..255 purely for visualization colormap HUD
        depth_norm = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

        return depth_map, depth_colored

    def get_object_distance(self, depth_map, bbox):
        """
        Calculates estimated distance (in meters) and hazard zone for bounding box [x, y, w, h].
        Fuses raw disparity with optical bounding box geometry:
          - Prevents relative min-max collapse in enclosed rooms.
          - Filters out background objects across the room (> 2.8m).
          - Checks ground-plane contact line (y + h).
        """
        x, y, w, h = bbox
        frame_h, frame_w = depth_map.shape[:2]

        # Clamp coordinates to frame
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(frame_w, int(x + w))
        y2 = min(frame_h, int(y + h))

        if x2 <= x1 or y2 <= y1:
            return 9.9, "SAFE", (0, 255, 0)

        # Sample inner 50% central region of the box to avoid background bleed
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        sample_roi = depth_map[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]

        if sample_roi.size == 0:
            sample_roi = depth_map[y1:y2, x1:x2]

        # 70th percentile of disparity inside the object
        disparity = float(np.percentile(sample_roi, 70))

        # 1. Absolute Disparity Distance (calibrated to MiDaS v2.1 Small scale)
        disp_dist = 1600.0 / max(disparity, 180.0)

        # 2. Geometric Perspective Distance
        h_frac = float(h) / max(frame_h, 1)
        bottom_y_frac = float(y + h) / max(frame_h, 1)
        geom_dist = 0.55 / max(h_frac, 0.08)

        # 3. Sensor Fusion (weighted blend)
        estimated_meters = round(0.65 * disp_dist + 0.35 * geom_dist, 1)

        # 4. Enclosed Room / Ground Plane Guard:
        # If an object's bottom edge is in the upper half of the frame (not near the floor),
        # or if it occupies less than 6% of the frame, it is on the far wall or ceiling.
        if bottom_y_frac < 0.55 or h_frac < 0.07:
            estimated_meters = max(estimated_meters, 3.2)

        # Bound realistic range
        estimated_meters = max(0.4, min(estimated_meters, 8.0))

        # Categorize
        if estimated_meters < 1.2 and bottom_y_frac >= 0.58:
            status = "CRITICAL"
            color = (0, 0, 255)      # Red
        elif estimated_meters < 2.5:
            status = "NEAR"
            color = (0, 255, 255)    # Yellow
        else:
            status = "SAFE"
            color = (0, 255, 0)      # Green

        return estimated_meters, status, color
