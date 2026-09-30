import curses

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


LEFT_THRUST_TOPIC = '/wamv/thrusters/left/thrust'
RIGHT_THRUST_TOPIC = '/wamv/thrusters/right/thrust'

THRUST = 50


class WamvTeleop(Node):

    def __init__(self):
        super().__init__('wamv_teleop')

        self.left_pub = self.create_publisher(
            Float64,
            LEFT_THRUST_TOPIC,
            10
        )

        self.right_pub = self.create_publisher(
            Float64,
            RIGHT_THRUST_TOPIC,
            10
        )

    def publish_thrust(self, left, right):
        left_msg = Float64()
        right_msg = Float64()

        left_msg.data = float(left)
        right_msg.data = float(right)

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

    def stop(self):
        self.publish_thrust(0.0, 0.0)


def teleop_loop(stdscr, node):

    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    stdscr.addstr(
        0, 0,
        'WAM-V Keyboard Teleop\n\n'
        'Arrow Up    : forward\n'
        'Arrow Down  : reverse\n'
        'Arrow Left  : rotate left\n'
        'Arrow Right : rotate right\n'
        'Space       : stop\n'
        'q           : quit\n\n'
        f'Thrust: {THRUST:.2f}\n'
    )

    stdscr.refresh()

    try:
        while rclpy.ok():

            key = stdscr.getch()

            if key == curses.KEY_UP:

                node.publish_thrust(
                    THRUST,
                    THRUST
                )

                status = 'FORWARD'

            elif key == curses.KEY_DOWN:

                node.publish_thrust(
                    -THRUST,
                    -THRUST
                )

                status = 'REVERSE'

            elif key == curses.KEY_LEFT:

                node.publish_thrust(
                    -THRUST,
                    THRUST
                )

                status = 'LEFT'

            elif key == curses.KEY_RIGHT:

                node.publish_thrust(
                    THRUST,
                    -THRUST
                )

                status = 'RIGHT'

            elif key == ord(' '):

                node.stop()
                status = 'STOP'

            elif key in (ord('q'), ord('Q')):

                node.stop()
                break

            else:
                continue

            stdscr.addstr(
                10, 0,
                f'Command: {status:<15}'
            )

            stdscr.refresh()

    finally:
        node.stop()


def main(args=None):

    rclpy.init(args=args)

    node = WamvTeleop()

    try:
        curses.wrapper(
            teleop_loop,
            node
        )

    except KeyboardInterrupt:
        node.stop()

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()