import os
import time
import numpy as np
import mujoco
import mujoco.viewer

def show_nominal_posture():
    print("🤖 [조각상 모드] 명목 자세(Nominal Posture)를 띄웁니다...")

    # 1. 기존과 동일하게 MuJoCo 모델 불러오기
    xml_path = os.path.expanduser("~/ros2_ws_py/src/mujoco_menagerie/franka_emika_panda/scene.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # 2. 궁금해하셨던 바로 그 '편안한 자세' (Franka Ready Pose)
    q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    # 3. 로봇의 관절 위치(qpos)에 이 각도들을 강제로 쑤셔 넣습니다.
    mj_data.qpos[:7] = q_nominal

    # 4. ★핵심★: mj_step(물리 엔진 진행) 대신 mj_forward(기구학 업데이트)만 실행!
    # 이렇게 하면 중력에 의해 로봇이 바닥으로 떨어지지 않고 딱 멈춰있습니다.
    mujoco.mj_forward(mj_model, mj_data)

    print("👀 창이 열리면 마우스로 이리저리 돌려보세요. (창을 닫으면 종료됩니다)")

    # 5. 뷰어 띄우기
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)  # 화면 갱신만 0.1초마다 가볍게 해줍니다.

if __name__ == '__main__':
    show_nominal_posture()