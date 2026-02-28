# -------import base package---------
import os
import time
import numpy as np

# -------import ros2 package---------
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import warnings

# --------import mujoco package--------
import mujoco
import mujoco.viewer
import pinocchio as pin
from qpsolvers import solve_qp

from robot_descriptions.panda_description import URDF_PATH 

# ---------Generate Trajectory----------
class MinJerkTrajectory:

    #---------Calculate Error between current and target---------
    def __init__(self, start_pos, start_quat, end_pos, end_quat, duration):
        self.p0 = np.array(start_pos)
        self.pf = np.array(end_pos)
        self.T = duration
        self.q0 = pin.Quaternion(start_quat).normalized()
        self.qf = pin.Quaternion(end_quat).normalized()
        
        # quaternian optimization
        if self.q0.dot(self.qf) < 0: 
            self.qf = pin.Quaternion(-self.qf.coeffs())
            
        self.q_diff = self.q0.inverse() * self.qf
        self.log_diff = pin.log3(self.q_diff.matrix()) 

    # --------Calculate Trajectory that robot would follow ----------
    def get_state(self, t):

        # publsih last target even though time is over
        # anyway at t = T, value published is same that return value.
        if t >= self.T: 
            return (self.pf, np.zeros(3), np.zeros(3), self.qf, np.zeros(3), np.zeros(3))
            
        tau = t / self.T
        tau2, tau3, tau4, tau5 = tau**2, tau**3, tau**4, tau**5
        s = 10*tau3 - 15*tau4 + 6*tau5
        s_dot = (30*tau2 - 60*tau3 + 30*tau4) / self.T
        s_ddot = (60*tau - 180*tau2 + 120*tau3) / (self.T**2)
        
        p_ref = self.p0 + (self.pf - self.p0) * s
        v_ref = (self.pf - self.p0) * s_dot
        a_ref = (self.pf - self.p0) * s_ddot
        
        current_log = self.log_diff * s
        q_relative = pin.Quaternion(pin.exp3(current_log))
        q_ref = self.q0 * q_relative

        #------local to world-----

        w_local = self.log_diff * s_dot
        dw_local = self.log_diff * s_ddot

        R = q_ref.matrix()
        w_ref = R @ w_local
        dw_ref = R @ dw_local
        
        return p_ref, v_ref, a_ref, q_ref, w_ref, dw_ref


class TsidAllInOneNode(Node):
    def __init__(self):
        super().__init__('tsid_controller_node')
        print("initiating robot model...")

        self.publish_rate = 30.0 
        self.last_pub_time = 0.0
        
        # MuJoCo & Pinocchio setting
        xml_path = os.path.expanduser("~/ros2_ws_py/src/mujoco_menagerie/franka_emika_panda/scene.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.pin_model = pin.buildModelFromUrdf(URDF_PATH)
        self.pin_data = self.pin_model.createData()
        self.n_pin, self.n_arm = self.pin_model.nq, 7
        
        # ROS Publishers & Broadcasters
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, 'target_marker', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Control Gain
        self.Kp_pos, self.Kd_pos = 250.0, 2.0 * np.sqrt(250.0)
        self.Kp_rot, self.Kd_rot = 250.0, 2.0 * np.sqrt(250.0)
        self.Kp_post, self.Kd_post = 125.0, 2.0 * np.sqrt(125.0)
        self.w_ee, self.w_post = 1.0, 0.007
        self.q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

        self.target_body_name = "panda_hand_tcp"
        self.mj_ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.pin_ee_id = self.pin_model.getFrameId(self.target_body_name) if self.pin_model.existFrame(self.target_body_name) else self.pin_model.nframes - 1

        self.traj = None
        self.start_time = 0.0
        self.move_duration = 3.0
        self.state = "IDLE"
        mujoco.mj_step(self.mj_model, self.mj_data)

    def generate_random_target(self):
        # 1. 위치: 로봇 앞쪽 공간으로 제한 (너무 멀거나 몸통 뒤로 안 가게)
        # x: 앞쪽 0.3 ~ 0.6m, y: 좌우 -0.3 ~ 0.3m, z: 높이 0.2 ~ 0.5m
        pos = np.array([np.random.uniform(0.3, 0.6), 
                        np.random.uniform(-0.2, 0.2),
                        np.random.uniform(0.2, 0.5)])

        # (1) 기본 자세: 바닥 보기 (w=0, x=1, y=0, z=0)
        q_home = pin.Quaternion(0, 1, 0, 0)

        # (2) 랜덤 각도 생성 (Roll, Pitch, Yaw)
        # - Roll (X축 회전): 좌우로 기울기 (+- 45도)
        # - Pitch (Y축 회전): 앞뒤로 까닥거리기 (+- 45도)
        # - Yaw (Z축 회전): 제자리 뱅글뱅글 (360도 자유)
        roll  = np.random.uniform(-0.39, 0.39) # 약 +/- 22.5도
        pitch = np.random.uniform(-0.39, 0.39)  # 약 +/- 22.6도
        yaw   = np.random.uniform(0, 1.57)  # +/- 90도 (완전 자유)

        # (3) RPY -> Rotation Matrix 변환 (Pinocchio 유틸리티 사용)
        rot_mat = pin.utils.rpyToMatrix(roll, pitch, yaw)
        
        # (4) 회전 행렬을 쿼터니언으로 변환
        q_rand = pin.Quaternion(rot_mat)
        
        # (5) 최종 자세 = (기본 자세) * (랜덤 회전)
        quat = q_home * q_rand
        
        return pos, quat

    def publish_ros_msgs(self, q_arm):
        now = self.get_clock().now().to_msg()
        
        # 1. World 좌표계 방송
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "world"
        t.child_frame_id = "panda_link0"
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        # 2. Joint State (여기 수정됨!)
        js = JointState()
        js.header.stamp = now
        
        # 기존 7개 관절 이름 + 손가락 2개 이름 추가
        js.name = [f"panda_joint{i+1}" for i in range(7)] + ["panda_finger_joint1", "panda_finger_joint2"]
        
        # 기존 7개 각도 + 손가락 벌림 정도(0.04m) 추가
        # (손가락은 prismatic joint라 미터 단위입니다. 0.04면 4cm 벌림)
        js.position = q_arm.tolist() + [0.04, 0.04] 
        
        self.joint_pub.publish(js)

        # 3. Marker
        if self.traj:
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = now
            marker.ns, marker.id, marker.type, marker.action = "target", 0, Marker.ARROW, Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = self.traj.pf[0], self.traj.pf[1], self.traj.pf[2]
            marker.pose.orientation.w, marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = self.traj.qf.w, self.traj.qf.x, self.traj.qf.y, self.traj.qf.z
            marker.scale.x, marker.scale.y, marker.scale.z = 0.15, 0.03, 0.03 
            marker.color.a, marker.color.r, marker.color.g = 1.0, 0.0, 1.0 
            self.marker_pub.publish(marker)

    def run_simulation(self):
        print("🚀 시뮬레이션 시작! (Ki 제어 추가됨)")
        
        # 1. 게인 설정 (I 게인 추가!)
        Ki_pos = 100.0   # 위치 오차를 잡는 적분 게인
        Ki_rot = 50.0    # 자세 오차를 잡는 적분 게인
        dt = self.mj_model.opt.timestep # 시뮬레이션 타임스텝 (보통 0.002초)

        # 오차 적분값 저장 변수 (누적 에러)
        integral_error_pos = np.zeros(3)
        integral_error_rot = np.zeros(3)

        # 오차 허용 범위
        TOL_POS = 0.005  # 0.5cm (더 엄격하게)
        TOL_ROT = 0.05   # 약 2.8도

        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()
                
                # --- 상태 업데이트 ---
                q_arm = self.mj_data.qpos[:self.n_arm].copy()
                v_arm = self.mj_data.qvel[:self.n_arm].copy()
                q_pin = np.zeros(self.n_pin); q_pin[:self.n_arm] = q_arm
                v_pin = np.zeros(self.n_pin); v_pin[:self.n_arm] = v_arm

                pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
                pin.updateFramePlacements(self.pin_model, self.pin_data)
                curr_pos = self.pin_data.oMf[self.pin_ee_id].translation
                curr_quat = pin.Quaternion(self.pin_data.oMf[self.pin_ee_id].rotation)

                # --- 상태 머신 ---
                if self.state == "IDLE":
                    # 타겟이 바뀔 때 적분 오차 초기화 (중요!)
                    integral_error_pos = np.zeros(3)
                    integral_error_rot = np.zeros(3)
                    
                    target_pos, target_quat = self.generate_random_target()
                    print(f"\n[New Target] Pos: {np.round(target_pos,2)}")
                    self.traj = MinJerkTrajectory(curr_pos, curr_quat, target_pos, target_quat, self.move_duration)
                    self.start_time = self.mj_data.time
                    self.state = "MOVING"

                elapsed = self.mj_data.time- self.start_time
                p_ref, v_ref, a_ref, q_ref, w_ref, dw_ref = self.traj.get_state(elapsed)

                # --- 오차 계산 및 검사 ---
                if self.state == "MOVING":
                    err_pos_norm = np.linalg.norm(self.traj.pf - curr_pos)
                    q_diff_check = self.traj.qf * curr_quat.inverse()
                    err_rot_norm = 2 * np.arccos(np.clip(abs(q_diff_check.w), -1.0, 1.0))

                    if elapsed >= self.move_duration:
                        if err_pos_norm < TOL_POS and err_rot_norm < TOL_ROT:
                            print(f" 도착 완료! (오차 - Pos: {err_pos_norm:.4f}m, Rot: {err_rot_norm:.4f}rad)")
                            self.state = "IDLE" # 바로 다음 타겟으로
                        else:
                            # 아직 수렴 못했으면 계속 제어하면서 기다림 
                            if int(elapsed * 10) % 5 == 0: # 로그 도배 방지
                                print(f"수렴 중... PosErr: {err_pos_norm:.3f}, RotErr: {err_rot_norm:.3f}")

                # --- TSID 제어 (I 게인 적용) ---
                pin.computeAllTerms(self.pin_model, self.pin_data, q_pin, v_pin)
                pin.updateFramePlacements(self.pin_model, self.pin_data)
                
                J = pin.getFrameJacobian(self.pin_model, self.pin_data, self.pin_ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, :self.n_arm]
                
                pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q_pin, v_pin)
                pin.updateFramePlacements(self.pin_model, self.pin_data)
                Jdot_full = pin.getFrameJacobianTimeVariation(
                    self.pin_model, 
                    self.pin_data, 
                    self.pin_ee_id, 
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )

                # 3. 전체 행렬 중 로봇 팔의 자유도(n_arm)만큼 슬라이싱 후 현재 속도(v_arm)와 행렬 곱셈
                Jdot = Jdot_full[:, :self.n_arm]

                dJV = Jdot @ v_arm
                curr_v_pin = pin.getFrameVelocity(self.pin_model, self.pin_data, self.pin_ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
                curr_w_pin = pin.getFrameVelocity(self.pin_model, self.pin_data, self.pin_ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).angular

                # 1. 위치 오차 (P, D)
                e_pos = p_ref - curr_pos
                e_vel = v_ref - curr_v_pin
                
                # 2. 자세 오차 (P, D)
                e_rot_vec = pin.log3((q_ref * curr_quat.inverse()).matrix())
                
                # 3. [핵심] 적분 오차 누적 (Anti-windup: 너무 커지면 제한)
                if self.state == "MOVING": # 움직일 때만 적분
                    integral_error_pos += e_pos * dt
                    integral_error_rot += e_rot_vec * dt
                    
                    # 적분값 제한 (폭주 방지, 최대 0.5 정도의 가속도 기여)
                    limit = 0.25
                    integral_error_pos = np.clip(integral_error_pos, -limit, limit)
                    integral_error_rot = np.clip(integral_error_rot, -limit, limit)

                # 4. 가속도 명령 생성 (PID 제어)
                # a_cmd = a_ref + (Kp * e) + (Kd * de) + (Ki * integral_e)
                a_pos = a_ref + (self.Kp_pos * e_pos) + (self.Kd_pos * e_vel) + (Ki_pos * integral_error_pos)
                a_rot = dw_ref + (self.Kp_rot * e_rot_vec) + (self.Kd_rot * (w_ref - curr_w_pin)) + (Ki_rot * integral_error_rot)
                
                b_acc = np.concatenate([a_pos, a_rot]) - dJV
                e_post = self.q_nominal - q_arm
                a_post = self.Kp_post * e_post - self.Kd_post * v_arm
                
                P = (self.w_ee * J.T @ J) + (self.w_post * np.eye(self.n_arm)) + (1e-4 * np.eye(self.n_arm))
                q_qp = -(self.w_ee * J.T @ b_acc) - (self.w_post * a_post)
                M = self.pin_data.M[:self.n_arm, :self.n_arm]
                h = self.pin_data.nle[:self.n_arm]
                
                ddq = solve_qp(P, q_qp, np.vstack([M, -M]), np.concatenate([self.tau_max - h, self.tau_max + h]), solver="osqp")
                if ddq is not None: self.mj_data.ctrl[:self.n_arm] = M @ ddq + h

                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync()

                current_sim_time = self.mj_data.time
                if current_sim_time - self.last_pub_time >= (1.0 / self.publish_rate):
                    self.publish_ros_msgs(q_arm)
                    self.last_pub_time = current_sim_time
                
                dt_step = time.time() - step_start
                if dt_step < self.mj_model.opt.timestep: time.sleep(self.mj_model.opt.timestep - dt_step)
def main():
    rclpy.init()
    try: TsidAllInOneNode().run_simulation()
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()