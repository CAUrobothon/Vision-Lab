from vision_interface import (Color, FakeVisionSource, SyntheticPixelToWorld, by_color,
                              closest_grasp_yaw, localize, nearest_first)

source = FakeVisionSource.random("B", seed=1)  # 대회 날: 비전 담당이 주는 CubevisionSource
p2w = SyntheticPixelToWorld()                  # 대회 날: 캘리브레이션 팀원의 PixelToWorld
gripper_yaw = 0.0                              # 지금 그리퍼 yaw (로봇 프레임, 도)

while True:
    scene = localize(source.get_scene(), p2w)  # 픽셀 -> mm. 팔이 큐브를 가리지 않을 때 부를 것
    if scene.is_synthetic:
        print("주의: 합성 좌표 - 실제 로봇에 쓰지 말 것")
    reds = [c for c in by_color(scene.cubes, Color.RED) if "partial" not in c.flags]
    if not reds:
        break
    cube = nearest_first(reds)[0]              # 프레임 원점에서 가까운 순
    p = cube.pose                              # 윗면 중심, 로봇 프레임 mm
    yaw = closest_grasp_yaw(p.yaw_deg, gripper_yaw) if p.yaw_deg is not None else gripper_yaw
    print(f"집기: ({p.x_mm:.1f}, {p.y_mm:.1f}, {p.z_mm:.1f}) mm, 그리퍼 yaw {yaw:.1f}°")
    # ... 여기서 IK -> 집기 -> 바구니에 놓기 ...
    gripper_yaw = yaw
    source.remove(cube)                        # 가짜 소스 전용: 집은 큐브를 장면에서 뺀다
