import os
import requests

# 1. 설정
package_dir = os.getcwd() # 현재 폴더 (franka_tsid_control)
mesh_dir = os.path.join(package_dir, "meshes")
urdf_path = os.path.join(package_dir, "panda_fixed.urdf")

# 공식 Franka ROS 저장소 URL
base_url = "https://raw.githubusercontent.com/frankaemika/franka_ros/develop/franka_description/meshes"

# 다운로드할 파일 목록
visual_files = [f"link{i}.dae" for i in range(8)] + ["hand.dae", "finger.dae"]
collision_files = [f"link{i}.stl" for i in range(8)] + ["hand.stl", "finger.stl"]

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"  - 이미 있음: {os.path.basename(save_path)}")
        return
    print(f"  - 다운로드 중... {os.path.basename(save_path)}")
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(r.content)
        else:
            print(f"    ❌ 실패 (Status {r.status_code}): {url}")
    except Exception as e:
        print(f"    ❌ 에러: {e}")

# 2. 폴더 생성 및 다운로드
print("📦 1. Mesh 파일 다운로드 시작...")
os.makedirs(os.path.join(mesh_dir, "visual"), exist_ok=True)
os.makedirs(os.path.join(mesh_dir, "collision"), exist_ok=True)

for f in visual_files:
    download_file(f"{base_url}/visual/{f}", os.path.join(mesh_dir, "visual", f))

for f in collision_files:
    download_file(f"{base_url}/collision/{f}", os.path.join(mesh_dir, "collision", f))

# 3. URDF 수정
print("\n📝 2. URDF 경로 수정 중...")

if not os.path.exists(urdf_path):
    print(f"❌ 에러: {urdf_path} 파일이 없습니다!")
    exit()

with open(urdf_path, 'r') as f:
    urdf_content = f.read()

# 기존의 긴 절대경로들을 현재 다운받은 경로로 교체
# (어떤 경로가 적혀있든, 파일명만 맞으면 우리 경로로 바꿔버림)
new_visual_path = f"file://{mesh_dir}/visual/"
new_collision_path = f"file://{mesh_dir}/collision/"

import re

# 정규표현식으로 경로 교체 (visual)
# mesh filename=".../visual/link0.dae" -> mesh filename="file://.../visual/link0.dae"
pattern_visual = r'filename=".*\/visual\/(.*\.dae)"'
urdf_content = re.sub(pattern_visual, f'filename="{new_visual_path}\\1"', urdf_content)

# 정규표현식으로 경로 교체 (collision)
pattern_collision = r'filename=".*\/collision\/(.*\.stl)"'
urdf_content = re.sub(pattern_collision, f'filename="{new_collision_path}\\1"', urdf_content)

# hand.dae가 없을 때를 대비해 stl로 되어있던 것을 다시 dae로 복구 (방금 다운받았으니)
urdf_content = urdf_content.replace('visual/hand.stl', 'visual/hand.dae')

# 결과 저장
with open(urdf_path, 'w') as f:
    f.write(urdf_content)

print("✅ 완료! URDF 파일이 수정되었습니다.")
print(f"   - 경로: {urdf_path}")
print("🚀 이제 다시 robot_state_publisher를 실행해보세요!")
