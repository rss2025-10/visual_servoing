#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation


class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        super().__init__("cone_detector")
        # toggle line follower vs cone parker
        self.LineFollower = False

        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.get_logger().info("Cone Detector Initialized")

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        try:
            # Convert ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # Create a binary mask to keep only the middle section of the image
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Define the middle section parameters (adjust these as needed)
            middle_width_percent = 1.0  # Keep middle 60% horizontally
            middle_height_percent = 0.3  # Keep middle 80% vertically
            
            # Calculate mask boundaries
            left_boundary = int(width * (1 - middle_width_percent) / 2)
            right_boundary = int(width * (1 - (1 - middle_width_percent) / 2))
            top_boundary = int(height * (1 - middle_height_percent) / 2)
            bottom_boundary = int(height * (1 - (1 - middle_height_percent) / 2))
            
            # Create the mask with the middle section set to 255 (white)
            mask[top_boundary:bottom_boundary, left_boundary:right_boundary] = 255
            
            # Apply the mask to the image
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            
            # Apply color segmentation to detect the cone
            (min_pt, max_pt) = cd_color_segmentation(masked_image)
            
            # Extract the bounding box coordinates
            xmin, ymin = min_pt
            xmax, ymax = max_pt
            
            # Calculate the center pixel at the bottom of the bounding box
            # This is the point that corresponds to the ground plane
            u = (xmin + xmax) / 2  # center x-coordinate
            v = float(ymax)  # bottom y-coordinate
            
            # Create and publish the cone location message
            cone_msg = ConeLocationPixel()
            cone_msg.u = u
            cone_msg.v = v
            self.cone_pub.publish(cone_msg)
            
            # Draw the bounding box and bottom center point on the debug image
            debug_img = image.copy()
            
            # Draw the mask boundaries for debugging
            height, width = image.shape[:2]
            left_boundary = int(width * (1 - middle_width_percent) / 2)
            right_boundary = int(width * (1 - (1 - middle_width_percent) / 2))
            top_boundary = int(height * (1 - middle_height_percent) / 2)
            bottom_boundary = int(height * (1 - (1 - middle_height_percent) / 2))
            
            # Blue rectangle showing the mask boundaries
            cv2.rectangle(debug_img, (left_boundary, top_boundary), (right_boundary, bottom_boundary), (255, 0, 0), 2)
            
            # Green bounding box for the cone
            cv2.rectangle(debug_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # Red circle at bottom center
            cv2.circle(debug_img, (int(u), int(v)), 5, (0, 0, 255), -1)
            
            # Publish the debug image
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_msg)
            
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"No cone detected: {e}")

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()