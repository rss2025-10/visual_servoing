#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np


from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic", "/drive")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_cone", 
            self.relative_cone_callback, 1)

        self.parking_distance = 0.5 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        self.car_length = 0.325

        self.get_logger().info("Parking Controller Initialized")

        self.backing_up = False

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        self.get_logger().info(f"[INFO] relative_x: {self.relative_x}, relative_y: {self.relative_y}")
        drive_cmd = AckermannDriveStamped()
        distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2)
        # if self.relative_x < 0:
        #     self.backing_up = True


        if self.backing_up:
            if self.relative_x < self.parking_distance:
                # Scale reverse speed proportionally to distance error
                # Minimum speed of -0.3 when close, up to -1.0 when far
                scale_factor = min(1.0, max(0.3, abs(self.relative_x - self.parking_distance)))
                drive_cmd.drive.speed = -float(scale_factor)
                drive_cmd.drive.steering_angle = float(0)
            else:
                self.backing_up = False
        else:   
            if abs(distance_error) < self.parking_distance:
                drive_cmd.drive.speed = float(0)
                self.get_logger().info(f'side error {abs(self.relative_x - self.relative_y):.2f}')
                if 0.55 > abs(self.relative_x - self.relative_y) or 0.85 < abs(self.relative_x - self.relative_y):
                    self.backing_up = True
            else:
                lookahead_distance = max(distance_error, 0.1) # no div by 0

                cone_vec = np.array([[self.relative_x, self.relative_y]]).T + np.array([[self.car_length, 0]]).T
                target = cone_vec
                np.linalg.norm(target)
                eta = np.arctan(target[1], target[0])
                # eta2 = abs(self.relative_y) / lookahead_distance
                # eta = eta1 + eta2 # assuming small angle approx.

                # steering_angle = np.arctan(2 * self.car_length * np.sin(eta), lookahead_distance)
                steering_angle = np.arctan(2 * self.car_length * np.sin(eta) / lookahead_distance)[0]
                drive_cmd.drive.steering_angle = 5* steering_angle
                
                # Scale forward speed based on distance error
                # Slower when closer to the target (0.3 minimum), up to 1.0 when far
                scale_factor = min(1.0, max(0.3, distance_error))
                drive_cmd.drive.speed = float(scale_factor)

        # self.get_logger().info(f'[INFO] angle: {steering_angle}')
        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)
        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2)

        #################################
        
        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()