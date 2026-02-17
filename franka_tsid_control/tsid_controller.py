# ----------import ros2 package---------- 
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
#-----------import rest package----------
import os
import mujoco
import mujoco.viewer
import time

#-----------ros2 code-------------
class MujocoSubscriberNode(Node):
    def __init__(self):
        super().__init__('mujoco_subscriber_node')

        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/panda/torque_commands',
            self.torque_callback,
            10
        )

        self.latest_torques = [0.0]*8

        #---------path setting---------
        xml_path = os.path.expanduser("~/ros2_ws_py/src/mujoco_menagerie/franka_emika_panda/scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    #--------receive torque input from publisher and update -------
    def torque_callback(self, msg):
        self.latest_torques = msg.data

    #--------launch mujoco and update torque input to robot-----
    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and rclpy.ok():

                rclpy.spin_once(self, timeout_sec=0)
                #------update torque input to mujoco-------
                for i in range(8):
                    self.data.ctrl[i] = self.latest_torques[i]
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)

def main(args=None):
    rclpy.init(args=args)
    node = MujocoSubscriberNode()

    print("mujoco simulation and ready for receiving torque input")
    node.run_simulation()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()