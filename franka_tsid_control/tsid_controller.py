import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# ==============================================================================
# [안전 장치] 패키지 에러 방지
# ==============================================================================
try:
    import pinocchio as pin
    from robot_descriptions.panda_description import URDF_PATH
    from qpsolvers import solve_qp
except ImportError as e:
    print(f"\n❌ 패키지 에러: {e}")
    print("터미널에 아래 명령어를 복사해서 붙여넣고 설치해주세요!")
    print("---------------------------------------------------------")
    print("python3 -m pip install pin robot_descriptions qpsolvers osqp")
    print("---------------------------------------------------------\n")
    exit()

class TsidAllInOneNode:
    def __init__(self):
        print("🤖 [초기화] MuJoCo 및 Pinocchio 모델을 불러오는 중...")
        
        # 1. MuJoCo 세팅
        xml_path = os.path.expanduser("~/ros2_ws_py/src/mujoco_menagerie/franka_emika_panda/scene.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # 2. Pinocchio 세팅
        self.pin_model = pin.buildModelFromUrdf(URDF_PATH)
        self.pin_data = self.pin_model.createData()
        
        # 제어할 End-Effector 프레임
        self.ee_frame_name = "panda_link8"
        self.ee_frame_id = self.pin_model.getFrameId(self.ee_frame_name)
        self.nv = 7 
        
        # ---------------------------------------------------------------------
        # [NEW] 3. 제어기 게인 세팅 (Task 1: EE 추종 / Task 2: 자세 유지)
        # ---------------------------------------------------------------------
        # (1) Main Task: End-Effector 추종 게인 (강하게!)
        self.Kp_ee = 170
        self.Kd_ee = 2.0 * np.sqrt(self.Kp_ee)
        self.w_ee = 1.0  # 가중치 1.0 (최우선 순위)

        # (2) Posture Task: 명목 자세 유지 게인 (살살 부드럽게)
        self.Kp_post = 100.0  
        self.Kd_post = 2.0 * np.sqrt(self.Kp_post)
        self.w_post = 0.012 # 가중치 0.01 (EE 추종에 방해 안 되게 1/100 수준으로)

        # (3) 명목 자세 (Nominal Posture) 정의 - Franka의 'Ready' 포즈
        self.q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

        # (4) 기타 설정
        self.lambda_damp = 1e-4
        self.tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

    def run_simulation(self):
        print("🚀 [실행] 시뮬레이션 시작! (이제 팔꿈치가 예쁘게 따라올 겁니다)")
        self.mj_data.mocap_quat[0] = np.array([0.0, 1.0, 0.0, 0.0])
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # -----------------------------------------------------------
                # [Step 1] 현재 상태 읽어오기
                # -----------------------------------------------------------
                nq = self.pin_model.nq
                nv_pin = self.pin_model.nv
                
                q = self.mj_data.qpos[:nq].copy()
                v = self.mj_data.qvel[:nv_pin].copy()

                # Mocap 목표 위치/방위 읽기
                mocap_pos = self.mj_data.mocap_pos[0]
                mocap_quat = self.mj_data.mocap_quat[0] 
                
                quat = pin.Quaternion(mocap_quat[0], mocap_quat[1], mocap_quat[2], mocap_quat[3])
                target_se3 = pin.SE3(quat.matrix(), mocap_pos)

                # -----------------------------------------------------------
                # [Step 2] Pinocchio 수학 계산
                # -----------------------------------------------------------
                pin.computeAllTerms(self.pin_model, self.pin_data, q, v)
                pin.updateFramePlacements(self.pin_model, self.pin_data)
                
                # Drift 가속도
                pin.forwardKinematics(self.pin_model, self.pin_data, q, v, np.zeros(nv_pin))
                a_drift = pin.getFrameAcceleration(self.pin_model, self.pin_data, self.ee_frame_id, pin.ReferenceFrame.LOCAL).vector

                current_se3 = self.pin_data.oMf[self.ee_frame_id]
                J_full = pin.getFrameJacobian(self.pin_model, self.pin_data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
                J = J_full[:, :self.nv] 
                v_current = pin.getFrameVelocity(self.pin_model, self.pin_data, self.ee_frame_id, pin.ReferenceFrame.LOCAL).vector

                # -----------------------------------------------------------
                # [Step 3] 목표 가속도 계산 (두 가지 태스크!)
                # -----------------------------------------------------------
                
                # [Task 1] End-Effector 추종 (빨간 구슬 따라가기)
                error_se3 = current_se3.actInv(target_se3)
                e_ee = pin.log6(error_se3).vector  
                a_ee_des = self.Kp_ee * e_ee + self.Kd_ee * (-v_current)
                b_acc = a_ee_des - a_drift 

                # [Task 2] Posture Maintaining (편한 자세 유지하기) [NEW!]
                # q_nominal 과 현재 q 의 차이를 줄이도록 당김
                e_post = self.q_nominal - q[:self.nv]
                a_post_des = self.Kp_post * e_post + self.Kd_post * (-v[:self.nv])

                # -----------------------------------------------------------
                # [Step 4] QP Solver 세팅 (가중치 적용)
                # -----------------------------------------------------------
                # 목적 함수: w_ee * || J*ddq - b_acc ||^2 + w_post * || ddq - a_post ||^2
                
                # P 행렬 (가중치 w_post 추가)
                P = (self.w_ee * J.T @ J) + (self.w_post * np.eye(self.nv)) + (self.lambda_damp * np.eye(self.nv))
                
                # q_qp 벡터 (가중치 w_post 추가)
                # 수식 전개: - (w_ee * J.T * b_acc) - (w_post * a_post)
                q_qp = -(self.w_ee * J.T @ b_acc) - (self.w_post * a_post_des)

                # 제약 조건 (모터 토크 한계)
                M = self.pin_data.M[:self.nv, :self.nv]
                h_nle = self.pin_data.nle[:self.nv]
                
                G = np.vstack([M, -M])
                h_qp = np.concatenate([self.tau_max - h_nle, self.tau_max + h_nle])

                ddq = solve_qp(P, q_qp, G, h_qp, solver="osqp")

                # -----------------------------------------------------------
                # [Step 5] 토크 인가
                # -----------------------------------------------------------
                if ddq is not None:
                    tau = M @ ddq + h_nle
                    self.mj_data.ctrl[:self.nv] = tau 
                else:
                    self.mj_data.ctrl[:self.nv] = np.zeros(self.nv)

                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync()
                
                time_until_next_step = self.mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def main():
    node = TsidAllInOneNode()
    node.run_simulation()

if __name__ == '__main__':
    main()