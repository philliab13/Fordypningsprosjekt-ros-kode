import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


# ============================================================
# ROS topics
# ============================================================

LEFT_TOPIC = (
    '/wamv/sensors/cameras/front_left_camera_sensor/'
    'optical/image_raw'
)

RIGHT_TOPIC = (
    '/wamv/sensors/cameras/front_right_camera_sensor/'
    'optical/image_raw'
)


# ============================================================
# VRX stereo calibration
# ============================================================

FX = 762.7223205566406
BASELINE = 0.20  # metres

DISPLAY_SCALE = 0.5


# ============================================================
# Obstacle extraction
# ============================================================

def find_nearby_obstacles(
    depth,
    min_depth,
    max_depth,
    minimum_area_fraction,
    morphology_kernel_size=5,
):
    """
    Find connected stereo-depth regions inside the specified distance range.

    Returns a list of dictionaries containing:
        x, y, width, height
        area
        median_depth
        mean_depth
        nearest_depth
    """

    valid = (
        np.isfinite(depth)
        & (depth >= min_depth)
        & (depth <= max_depth)
    )

    mask = valid.astype(np.uint8) * 255

    # --------------------------------------------------------
    # Remove isolated disparity noise and fill small holes
    # --------------------------------------------------------

    if morphology_kernel_size > 1:

        kernel = np.ones(
            (morphology_kernel_size, morphology_kernel_size),
            dtype=np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    minimum_area = depth.size * minimum_area_fraction

    obstacles = []

    # Label 0 is background
    for label in range(1, count):

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]

        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        area = stats[label, cv2.CC_STAT_AREA]

        if area < minimum_area:
            continue

        # Only use pixels belonging to this component
        component_mask = labels == label

        component_depth = depth[component_mask]

        component_depth = component_depth[
            np.isfinite(component_depth)
        ]

        if component_depth.size == 0:
            continue

        median_depth = float(
            np.median(component_depth)
        )

        mean_depth = float(
            np.mean(component_depth)
        )

        # A robust-ish estimate of the nearest part of the obstacle.
        # More useful than the absolute minimum, which can be noisy.
        nearest_depth = float(
            np.percentile(component_depth, 10)
        )

        obstacles.append(
            {
                'x': int(x),
                'y': int(y),
                'width': int(width),
                'height': int(height),
                'area': int(area),

                'median_depth': median_depth,
                'mean_depth': mean_depth,
                'nearest_depth': nearest_depth,
            }
        )

    # Nearest obstacles first
    obstacles.sort(
        key=lambda obstacle: obstacle['median_depth']
    )

    return obstacles, mask


# ============================================================
# ROS node
# ============================================================

class StereoViewer(Node):

    def __init__(self):

        super().__init__('stereo_viewer')

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        self.obstacle_distance = self.declare_parameter(
            'obstacle_distance',
            10.0
        ).value

        self.minimum_depth = self.declare_parameter(
            'minimum_depth',
            0.10
        ).value

        self.minimum_obstacle_area_fraction = self.declare_parameter(
            'minimum_obstacle_area_fraction',
            0.01
        ).value

        self.morphology_kernel_size = self.declare_parameter(
            'morphology_kernel_size',
            5
        ).value

        # ----------------------------------------------------
        # Validate parameters
        # ----------------------------------------------------

        if (
            not np.isfinite(self.obstacle_distance)
            or self.obstacle_distance <= 0.0
        ):
            raise ValueError(
                'obstacle_distance must be finite and positive'
            )

        if (
            not np.isfinite(self.minimum_depth)
            or self.minimum_depth < 0.0
        ):
            raise ValueError(
                'minimum_depth must be finite and non-negative'
            )

        if self.minimum_depth >= self.obstacle_distance:
            raise ValueError(
                'minimum_depth must be smaller than obstacle_distance'
            )

        if not (
            0.0
            <= self.minimum_obstacle_area_fraction
            <= 1.0
        ):
            raise ValueError(
                'minimum_obstacle_area_fraction must be in [0, 1]'
            )

        self.morphology_kernel_size = int(
            self.morphology_kernel_size
        )

        if self.morphology_kernel_size < 1:
            self.morphology_kernel_size = 1

        # OpenCV kernels are nicer with odd sizes
        if self.morphology_kernel_size % 2 == 0:
            self.morphology_kernel_size += 1

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.bridge = CvBridge()

        self.frame_count = 0

        self.current_depth = None
        self.current_disparity = None

        self.clicked_point = None
        self.clicked_depth = None
        self.clicked_disparity = None

        # ----------------------------------------------------
        # ROS image subscribers
        # ----------------------------------------------------

        self.left_sub = Subscriber(
            self,
            Image,
            LEFT_TOPIC,
            qos_profile=qos_profile_sensor_data
        )

        self.right_sub = Subscriber(
            self,
            Image,
            RIGHT_TOPIC,
            qos_profile=qos_profile_sensor_data
        )

        self.sync = ApproximateTimeSynchronizer(
            [
                self.left_sub,
                self.right_sub
            ],
            queue_size=10,
            slop=0.02
        )

        self.sync.registerCallback(
            self.stereo_callback
        )

        # ----------------------------------------------------
        # StereoSGBM
        # ----------------------------------------------------

        num_disparities = 256
        block_size = 5

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,

            P1=8 * block_size ** 2,
            P2=32 * block_size ** 2,

            disp12MaxDiff=1,
            uniquenessRatio=10,

            speckleWindowSize=100,
            speckleRange=2,

            preFilterCap=63,

            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # ----------------------------------------------------
        # GUI
        # ----------------------------------------------------

        cv2.namedWindow(
            'VRX obstacle view'
        )

        cv2.setMouseCallback(
            'VRX obstacle view',
            self.mouse_callback
        )

        self.get_logger().info(
            'Stereo obstacle viewer started'
        )

        self.get_logger().info(
            f'fx={FX:.3f} px | '
            f'baseline={BASELINE:.3f} m | '
            f'obstacle threshold={self.obstacle_distance:.1f} m'
        )

    # ========================================================
    # Mouse depth measurement
    # ========================================================

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.current_depth is None:
            return

        # Display image is half size
        u = int(
            x / DISPLAY_SCALE
        )

        v = int(
            y / DISPLAY_SCALE
        )

        height, width = self.current_depth.shape

        if not (
            0 <= u < width
            and 0 <= v < height
        ):
            return

        # ----------------------------------------------------
        # Use small region rather than single noisy pixel
        # ----------------------------------------------------

        radius = 3

        x0 = max(
            0,
            u - radius
        )

        x1 = min(
            width,
            u + radius + 1
        )

        y0 = max(
            0,
            v - radius
        )

        y1 = min(
            height,
            v + radius + 1
        )

        depth_region = self.current_depth[
            y0:y1,
            x0:x1
        ]

        disparity_region = self.current_disparity[
            y0:y1,
            x0:x1
        ]

        valid = (
            np.isfinite(depth_region)
            & (depth_region > 0.0)
            & (disparity_region > 0.0)
        )

        self.clicked_point = (
            u,
            v
        )

        if not np.any(valid):

            self.clicked_depth = None
            self.clicked_disparity = None

            self.get_logger().info(
                f'Pixel ({u}, {v}): '
                f'no valid stereo depth'
            )

            return

        self.clicked_depth = float(
            np.median(
                depth_region[valid]
            )
        )

        self.clicked_disparity = float(
            np.median(
                disparity_region[valid]
            )
        )

        self.get_logger().info(
            f'Pixel ({u}, {v}) | '
            f'disparity={self.clicked_disparity:.2f} px | '
            f'depth={self.clicked_depth:.2f} m'
        )

    # ========================================================
    # Main callback
    # ========================================================

    def stereo_callback(
        self,
        left_msg,
        right_msg
    ):

        # ----------------------------------------------------
        # ROS -> OpenCV
        # ----------------------------------------------------

        left = self.bridge.imgmsg_to_cv2(
            left_msg,
            desired_encoding='bgr8'
        )

        right = self.bridge.imgmsg_to_cv2(
            right_msg,
            desired_encoding='bgr8'
        )

        left_gray = cv2.cvtColor(
            left,
            cv2.COLOR_BGR2GRAY
        )

        right_gray = cv2.cvtColor(
            right,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Stereo disparity
        #
        # StereoSGBM uses fixed-point disparity * 16
        # ----------------------------------------------------

        disparity = (
            self.stereo.compute(
                left_gray,
                right_gray
            ).astype(
                np.float32
            )
            / 16.0
        )

        # ----------------------------------------------------
        # Metric depth
        #
        # Z = f * B / d
        # ----------------------------------------------------

        depth = np.full(
            disparity.shape,
            np.nan,
            dtype=np.float32
        )

        valid_disparity = (
            disparity > 0.0
        )

        depth[
            valid_disparity
        ] = (
            FX
            * BASELINE
            / disparity[
                valid_disparity
            ]
        )

        self.current_depth = depth
        self.current_disparity = disparity

        # ----------------------------------------------------
        # Extract close obstacle regions
        # ----------------------------------------------------

        obstacles, obstacle_mask = (
            find_nearby_obstacles(
                depth=depth,

                min_depth=
                    self.minimum_depth,

                max_depth=
                    self.obstacle_distance,

                minimum_area_fraction=
                    self.minimum_obstacle_area_fraction,

                morphology_kernel_size=
                    self.morphology_kernel_size,
            )
        )

        # ----------------------------------------------------
        # Annotated camera image
        # ----------------------------------------------------

        left_annotated = left.copy()

        for obstacle in obstacles:

            x = obstacle['x']
            y = obstacle['y']

            width = obstacle['width']
            height = obstacle['height']

            median_depth = obstacle[
                'median_depth'
            ]

            nearest_depth = obstacle[
                'nearest_depth'
            ]

            cv2.rectangle(
                left_annotated,

                (
                    x,
                    y
                ),

                (
                    x + width - 1,
                    y + height - 1
                ),

                (
                    0,
                    0,
                    255
                ),

                3
            )

            label = (
                f'{median_depth:.2f} m '
                f'(near {nearest_depth:.2f} m)'
            )

            cv2.putText(
                left_annotated,

                label,

                (
                    max(
                        0,
                        x
                    ),
                    max(
                        30,
                        y - 10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (
                    0,
                    0,
                    255
                ),

                2,

                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Global status
        # ----------------------------------------------------

        if obstacles:

            nearest_obstacle = min(
                obstacle['median_depth']
                for obstacle in obstacles
            )

            status = (
                f'WARNING: '
                f'{len(obstacles)} obstacle(s) '
                f'< {self.obstacle_distance:g} m | '
                f'nearest ~ {nearest_obstacle:.2f} m'
            )

            status_colour = (
                0,
                0,
                255
            )

        elif not np.any(
            np.isfinite(depth)
        ):

            status = (
                'No valid stereo depth'
            )

            status_colour = (
                0,
                255,
                255
            )

        else:

            status = (
                f'No obstacle detected '
                f'< {self.obstacle_distance:g} m'
            )

            status_colour = (
                0,
                255,
                0
            )

        cv2.putText(
            left_annotated,

            status,

            (
                30,
                60
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            status_colour,

            2,

            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Clicked depth marker
        # ----------------------------------------------------

        if self.clicked_point is not None:

            u, v = self.clicked_point

            cv2.circle(
                left_annotated,

                (
                    u,
                    v
                ),

                8,

                (
                    0,
                    255,
                    255
                ),

                2
            )

            if self.clicked_depth is not None:

                click_text = (
                    f'{self.clicked_depth:.2f} m '
                    f'({self.clicked_disparity:.1f} px)'
                )

            else:

                click_text = (
                    'No valid depth'
                )

            text_x = min(
                u + 15,
                left_annotated.shape[1] - 280
            )

            text_y = max(
                v - 15,
                30
            )

            cv2.putText(
                left_annotated,

                click_text,

                (
                    text_x,
                    text_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (
                    0,
                    255,
                    255
                ),

                2,

                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Disparity visualization
        # ----------------------------------------------------

        disparity_display = np.zeros(
            disparity.shape,
            dtype=np.uint8
        )

        valid_disp = (
            disparity > 0.0
        )

        if np.any(
            valid_disp
        ):

            valid_values = disparity[
                valid_disp
            ]

            d_min = float(
                valid_values.min()
            )

            d_max = float(
                valid_values.max()
            )

            if d_max > d_min:

                disparity_display[
                    valid_disp
                ] = (
                    255.0
                    * (
                        valid_values
                        - d_min
                    )
                    / (
                        d_max
                        - d_min
                    )
                ).astype(
                    np.uint8
                )

        disparity_colour = (
            cv2.applyColorMap(
                disparity_display,
                cv2.COLORMAP_TURBO
            )
        )

        # ----------------------------------------------------
        # Resize GUI images
        # ----------------------------------------------------

        obstacle_view = cv2.resize(
            left_annotated,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE
        )

        left_small = cv2.resize(
            left,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE
        )

        right_small = cv2.resize(
            right,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE
        )

        disparity_small = cv2.resize(
            disparity_colour,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE
        )

        mask_small = cv2.resize(
            obstacle_mask,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_NEAREST
        )

        stereo_pair = cv2.hconcat(
            [
                left_small,
                right_small
            ]
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            'VRX obstacle view',
            obstacle_view
        )

        cv2.imshow(
            'VRX stereo pair',
            stereo_pair
        )

        cv2.imshow(
            'VRX disparity',
            disparity_small
        )

        cv2.imshow(
            f'VRX < {self.obstacle_distance:g}m mask',
            mask_small
        )

        cv2.waitKey(1)

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        self.frame_count += 1

        if (
            self.frame_count
            % 30
            == 0
        ):

            t_left = (
                left_msg.header.stamp.sec
                + left_msg.header.stamp.nanosec
                * 1e-9
            )

            t_right = (
                right_msg.header.stamp.sec
                + right_msg.header.stamp.nanosec
                * 1e-9
            )

            dt_ms = abs(
                t_left
                - t_right
            ) * 1000.0

            if obstacles:

                nearest = min(
                    obstacle[
                        'median_depth'
                    ]
                    for obstacle
                    in obstacles
                )

                self.get_logger().info(
                    f'Frames={self.frame_count} | '
                    f'dt={dt_ms:.3f} ms | '
                    f'obstacles={len(obstacles)} | '
                    f'nearest={nearest:.2f} m'
                )

            else:

                self.get_logger().info(
                    f'Frames={self.frame_count} | '
                    f'dt={dt_ms:.3f} ms | '
                    f'no obstacle < '
                    f'{self.obstacle_distance:g} m'
                )


# ============================================================
# Main
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = StereoViewer()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()