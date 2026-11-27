# vision_interface — 비전 · 캘리브레이션 · 기구학 사이의 약속

로보톤 Pick & Place(2026-11-29)에서 비전이 내놓는 데이터의 형식과, 그 데이터를 로봇 좌표(mm)로 바꾸는 자리를 정한 코드다.
표준 라이브러리만 쓴다(Python ≥ 3.10, OpenCV · numpy 불필요).

## 0. 역할 분담 — 먼저 읽기

| 담당 | 하는 일 | 내놓는 것 | 이 패키지의 자리 |
|---|---|---|---|
| **비전** (이 패키지) | 카메라 프레임에서 큐브 찾기 | **픽셀만**: 윗면 중심 · 네 꼭짓점(px), 이미지 각도(°, 90° 대칭), 색, 신뢰도, 플래그 | `VisionSource.get_scene()` → `Scene` (모든 `pose` = `None`) |
| **캘리브레이션** 팀원 | 픽셀 → **로봇 베이스 좌표(mm)** 변환, 그 좌표계 정의 | `PixelToWorld` 구현 1개 | 계약 `protocols.PixelToWorld`, 빈 템플릿 `homography.HomographyPixelToWorld` |
| **기구학** 팀원 | mm 좌표로 IK · 동작 · 전략 | — | `localize(scene, p2w)` 뒤 `cube.pose` 사용 |

- 비전은 로봇 베이스가 어디 있는지 모른다. 그래서 이 패키지의 기본 경로는 **mm를 하나도 만들지 않는다.**
- mm는 `localize(scene, pixel_to_world)`가 캘리브레이션 팀원의 `PixelToWorld`를 받았을 때만 채운다. 그때 `scene.localized_by`에 그 변환의 이름이 남는다.
- 로봇 베이스 좌표계(원점 · 축 방향 · z 기준)는 **캘리브레이션 팀원이 정한다** (열린 질문 C1, C2). 아래 4.2절의 규정집 좌표계는 참고용이다.
- 테스트용 mm는 `SyntheticPixelToWorld`로만 만들 수 있고, 이름이 `SYNTHETIC`으로 시작해 `scene.is_synthetic == True`가 된다.

```
카메라 ─► cubevision 파이프라인 ─► Scene(픽셀) ──► localize(scene, p2w) ──► Scene(픽셀 + pose mm) ─► IK · 동작
          [비전]                    get_scene()     [캘리브레이션의 PixelToWorld]                    [기구학]
```

## 1. 설치

- 복사: `vision_interface/` 폴더 하나만 프로젝트에 복사하면 된다(의존성 없음).
- 또는 `pip install -e ~/Desktop/로보톤/vision-lab/interface`

```python
from vision_interface import Scene, Cube, CubePose, Color, Mission, VisionSource, PixelToWorld, localize
```

## 2. 데이터 구조 (`vision_interface/scene.py`)

모두 frozen dataclass(읽기 전용)다. `==`로 비교되고, JSON으로 오가도 값이 그대로다(10절).

**`Scene`** — 카메라 프레임 1장의 결과

| 필드 | 단위 | 뜻 |
|---|---|---|
| `frame_id` | 정수 | 소스마다 1씩 증가. 소스끼리는 비교 불가 |
| `timestamp_s` | 초 | 프레임을 받은 시각, `time.time()` (Unix) |
| `image_size_px` | px | `(가로, 세로)` — 픽셀 좌표가 가리키는 프레임 크기 |
| `cubes` | — | `Cube` 튜플 |
| `mission` | — | `Mission.A` / `Mission.B` / `None` |
| `source` | — | 픽셀을 만든 쪽, 예: `"cubevision:cv.hsv_ranges"`, `"fake:random"` |
| `localized_by` | — | pose를 채운 `PixelToWorld.name`. `None` = 픽셀만 |
| `notes` | — | 비전 경고 문구 |

규칙: 장면은 "픽셀만"(모든 `pose` = `None`, `localized_by` = `None`)이거나 "하나의 캘리브레이션으로 전부 위치 계산됨"(모든 `pose` 있음, `localized_by` 있음) 둘 중 하나다. 섞이면 생성 시 `ValueError`.

**`Cube`** — 큐브 1개, **윗면** 기준

| 필드 | 단위 | 뜻 |
|---|---|---|
| `color` | — | `Color.RED` · `GREEN` · `BLUE` · `WOOD` · `UNKNOWN` (값은 `"red"` 등 cubevision과 같은 문자열) |
| `center_px` | px | 윗면 중심 `(u, v)` — 판 위 25 mm 높이의 점 (4.3절) |
| `corners_px` | px | 윗면 네 꼭짓점, 화면에서 시계 방향, 첫 점 = u+v가 가장 작은 꼭짓점. 모르면 `None` |
| `image_angle_deg` | ° | 윗면 한 변의 방향, +u에서 +v 쪽(화면 시계 방향)으로 잼, [0, 90). 모르면 `None`. **로봇 각도 아님** |
| `area_px2` | px² | 윗면 넓이 |
| `score` | 0–1 | 검출 신뢰도 |
| `color_score` | 0–1 | 색 판정 신뢰도 |
| `flags` | — | 경고 문자열 (아래 표) |
| `pose` | — | `CubePose` 또는 `None` (localize 전) |

**`CubePose`** — 로봇 프레임 위치 (`localize`만 만든다)

| 필드 | 단위 | 뜻 |
|---|---|---|
| `x_mm`, `y_mm`, `z_mm` | mm | 윗면 **중심**, 캘리브레이션 팀원이 정한 로봇 프레임 |
| `yaw_deg` | ° | 윗면 한 변의 방향, 그 프레임의 +x에서 +y 쪽으로, [0, 90). 모르면 `None` |
| `grasp_yaw_deg` | ° | `yaw_deg + k·90` 중 `bearing_deg`에 가장 가까운 값 (5절) |
| `r_mm`, `bearing_deg` | mm, ° | (계산 속성) 원점에서의 거리, `atan2(y, x)` |

**`flags`** (cubevision `postprocess.py`가 붙이는 값)

| 값 | 뜻 | 기구학에서 |
|---|---|---|
| `partial` | 일부가 가려짐(팔 등) | 위치가 덜 정확, 다시 보기 권장 |
| `touches_border` | 영상 가장자리에 닿음 | 위치가 덜 정확할 수 있음 |
| `split` | 여러 큐브 덩어리에서 잘라냄 | 붙어 있는 큐브 |
| `large_face` | 큐브 1개 윗면보다 큼 | 탑 꼭대기이거나 크기 설정 문제 |
| `side_band`, `shade_edge` | 옆면 띠 · 그림자 경계를 제거함 | 정보용 |
| `class_mismatch` | 모델 클래스와 픽셀 색이 다름 | 색을 의심 |
| `from_box`, `box_only` | 축 정렬 박스에서 윗면을 추정 | 각도 · 꼭짓점이 덜 정확 |

## 3. 계약 두 개 (`vision_interface/protocols.py`)

```python
class VisionSource(Protocol):          # 비전이 구현, 기구학이 호출
    def get_scene(self) -> Scene: ...  # 픽셀만. 호출 뒤에 찍은 새 프레임으로 검출하고 돌려줌(그동안 블록)

class PixelToWorld(Protocol):          # 캘리브레이션 팀원이 구현
    name: str                          # 이 캘리브레이션 설명 → Scene.localized_by
    def pixel_to_world(self, u_px: float, v_px: float, height_mm: float) -> tuple[float, float, float]: ...
```

`pixel_to_world`의 입력과 출력:

- 입력 `(u_px, v_px)`: 비전과 **같은 프레임**(같은 해상도, 같은 왜곡 보정)의 픽셀.
- 입력 `height_mm`: 그 픽셀이 보는 점이 **경기장 판 윗면에서 몇 mm 위인지**. 판 윗면은 규정집이 정한 유일한 높이 기준이다(§1.1, p.1: z = 0은 블록을 놓는 판 윗면). 판 위에 선 큐브의 윗면 중심은 `25.0`(`rules.CUBE_TOP_HEIGHT_MM`).
- 출력 `(x_mm, y_mm, z_mm)`: 캘리브레이션 팀원의 로봇 프레임 좌표. z까지 돌려주므로 기구학은 판 윗면이 어디인지 몰라도 된다.
- 못 바꾸는 높이는 `ValueError`.

## 4. 좌표 · 단위 · 평면

### 4.1 이미지 좌표 (비전 출력)

```
(0,0) ─────────────────────► u (px, 오른쪽)
  │
  │      c0 ●─────────● c1       corners_px: 화면에서 시계 방향
  │         │         │                     c0 = u+v가 가장 작은 꼭짓점
  │         │    ✚    │          ✚ center_px: 윗면 중심
  │         │         │
  │      c3 ●─────────● c2       image_angle_deg: 한 변이 +u에서 +v 쪽으로
  ▼                                                (화면 시계 방향) 돈 각도, [0, 90)
  v (px, 아래쪽)
```

### 4.2 규정집 좌표계 (참고 — 로봇 프레임 결정은 캘리브레이션 팀원 몫)

규정집 §1.1(p.1)과 도면 1/3 주기 1: 원점 (0, 0) = 로봇 베이스 회전축 중심, +x = 오른쪽, +y = 기둥 쪽, z = 0 = 경기장 판 윗면, 단위 mm.
카메라가 z = +601 위에 있으므로(§1.3) z는 위쪽, 즉 오른손 좌표계다. 각도는 원점 기준으로 +x에서 +y 쪽(위에서 보면 반시계)으로 잰다(도면 2/3 주기 2 · 5).

```
                               +y  (기둥 쪽)
                                ▲      기둥 100.3 × 65.3 (§1.4)
                  120°          │          60°
                     ╲   A: 구역2 (탑, 60–120°)  ╱
                      ╲         │         ╱      A: 구역1 오른쪽 (0–60°)
   A: 구역1 왼쪽        ╲        │        ╱       B: R 바구니 50–90°
      (120–180°)        ╲       │       ╱           G 바구니 20–50°
   B: 블록 배치 구역       ╲      │      ╱            B 바구니  0–20°
      (90–180°)            ╲ R90 │     ╱
  180° ◄──────────────────────── ● ─────────────────────────► +x   0°
                        원점 = 로봇 베이스 회전축 (사다리꼴 홈, 도면 1/3)
       모든 구역 반경 R90 – R250 (§1.2), 경기장 외곽 R250.5
```

`rules.py`의 구역(`MISSION_A_ZONE1`, `MISSION_B_BASKETS`, `zone_of` …)은 **이 규정집 좌표계** 값이다. `PixelToWorld`의 출력이 이 좌표계일 때만 그대로 쓸 수 있다(C1).

### 4.3 어느 평면의 픽셀인가 — 꼭 읽기

```
              ● 카메라 렌즈 (판 윗면 위 601 mm, §1.3 p.1, 수직 아래를 봄)
             ╱
            ╱    center_px가 가리키는 점 = 큐브 윗면 중심 (판 윗면 위 25 mm)
       ┌───●┐
       │    │ 25 mm 큐브 (§3 p.5)
  ═════╧════╧═══●═══════════ 경기장 판 윗면 (z = 0, §1.1)  ← 같은 픽셀을 z = 0 평면으로 보내면 여기(●)
  ─────────────────────────── 책상 (판 두께 17.5 mm, §1.6)
```

- 비전의 `center_px` · `corners_px`는 **큐브 윗면**, 즉 판 윗면 위 25 mm 평면의 점이다. 큐브가 판에 닿는 바닥면이 아니다.
- 판 윗면(z = 0)에 맞춘 호모그래피를 그대로 쓰면, 같은 광선이 판까지 더 내려가므로 카메라 바로 아래 점에서 601/576 = 1.043배, 약 **4.3 % 더 먼 곳**이 나온다. 그 점에서 100 mm 떨어진 큐브는 약 4 mm, 경기장 가장자리는 약 13 mm 어긋난다.
- 그래서 `PixelToWorld`는 높이를 인자로 받는다. 비전은 판 위 큐브에 `height_mm = 25`, 탑 꼭대기에 `25·n`(`rules.tower_top_height_mm(n)`), 바구니 바닥(판보다 4 mm 높음, 도면 3/3 B)에 놓인 큐브에 `29`를 쓴다.
- `localize(scene, p2w)`는 모든 큐브를 25 mm로 본다. 더 높은 큐브는 `localize_cube(cube, p2w, height_mm)`로 다시 계산한다. 비전은 큐브 크기로 층수를 믿을 만하게 구분하지 못하므로(층 사이 차이 3–6 px) 층수는 기구학의 작업 상태에서 가져온다.

## 5. 90° 대칭과 집기 각도

- 윗면은 정사각형이라 90°마다 같은 모양이다. 그래서 `yaw_deg`는 [0, 90)으로 접혀 있고, `yaw`, `yaw+90`, `yaw+180`, `yaw+270`은 **모두 같은 자세**다. 89°와 1°는 2° 차이다(88° 아님).
- 평행 그리퍼는 180°마다 같으므로 집을 수 있는 그리퍼 yaw는 `{yaw + k·90}` 전부다. 그리퍼 yaw를 집게가 닫히는 축으로 재든 손가락 면으로 재든 둘은 90° 차이라 같은 집합이 된다.
- `closest_grasp_yaw(cube_yaw, reference)` → 그 집합 중 `reference`에서 [−45°, 45°) 안의 값. `reference`에 지금 그리퍼 yaw를 넣으면 가장 적게 돌리는 각이 나온다. 정확히 45° 차이면 작은 쪽.
- `pose.grasp_yaw_deg`는 `reference = bearing_deg`(원점 → 큐브 방향)로 고른 값이다. 프레임 원점이 베이스 회전축이면 `grasp_yaw_deg − bearing_deg`가 "방사 방향으로 내려가는 자세"에서 필요한 최소 손목 회전이다. 손목 관절의 영점과 부호는 기구학이 맞춘다(K1).
- `image_angle_deg`는 화면 기준(시계 방향)이라 로봇에 쓰면 안 된다. 영상은 로봇 프레임에 대해 돌아가 있거나 뒤집혀 있을 수 있다. `localize`는 꼭짓점 네 개를 로봇 프레임으로 옮긴 뒤 거기서 `yaw_deg`를 잰다.

## 6. 사용 예 — 기구학 쪽 (`examples/pick_red_fake.py`, 테스트에서 실제로 돌림)

```python
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
```

실행: `cd ~/Desktop/로보톤/vision-lab/interface && python examples/pick_red_fake.py` (설치하지 않았다면 `PYTHONPATH=.`을 앞에 붙인다).

도움 함수 (`geometry.py`): `localize`, `localize_cube`, `fold90`, `closest_grasp_yaw`, `by_color(cubes, *colors)`, `nearest_first(cubes, x_mm=0, y_mm=0)`.
규정 값 (`rules.py`): `MISSION_CUBES`, `tower_target_colors(["red","blue","green"])` → 9층 목표 색, `tower_top_height_mm(n)`, `BASKET_POINTS`, `zone_of(x, y, mission)` 등.

## 7. 가짜 소스 (카메라 없이 개발)

```bash
python -m vision_interface.fake --mission A --seed 1                     # 장면 1개를 JSON으로 출력 (픽셀만)
python -m vision_interface.fake --mission B --seed 1 --count 5 --out scenes.json
python -m vision_interface.fake --mission B --seed 1 --localize-synthetic  # 합성 mm까지 채움 (SYNTHETIC 표시)
```

```python
FakeVisionSource.random("A", seed=3)          # 무작위 배치 1개를 유지. get_scene()마다 frame_id·시각만 새로. remove(cube)로 집기 흉내
FakeVisionSource.from_json_file("scenes.json")  # 기록된 장면을 순서대로 반복 재생 (loop=False면 마지막 장면 유지)
FakeVisionSource([scene1, scene2])            # 직접 만든 장면
```

규정을 따르는 부분: 큐브 수 · 색 · 배치 순서(미션 A: R R R G G G B B B, 미션 B: + 나무 3개, §5 p.7), 시작 구역 안(미션 A 구역 1 = 0–60°·120–180°, §3 p.5 / 미션 B 90–180°, §4 p.6), 큐브 전체가 구역 안, 겹치지 않음, 방향은 규정에 없으므로 [0, 90) 무작위.
1–20번 위치 표시(§5 p.7)는 당일 공개된다. 공개되면 `random_scene(..., positions_mm=[...])`에 넣으면 규정대로 중복 없이 뽑아 R, G, B, 나무 순으로 놓는다.

합성인 부분: 픽셀 위치는 `SyntheticPixelToWorld`(정사영, 0.5 mm/px, 원점 = 픽셀 (640, 700), 영상 위쪽 = +y)로 그린 것이라 실제 카메라에서 보일 자리가 아니다. 원근 · 검출 잡음 · 기둥 근처 금지 구역은 없고 점수는 모두 1.0이다.

## 8. 비전 쪽 연결 (비전 담당용, `cubevision_adapter.py`)

```python
import cv2
from cubevision.registry import get_pipeline
from vision_interface import CubevisionSource

cap = cv2.VideoCapture(0)
pipe = get_pipeline("cv.hsv_ranges")
def grab():
    for _ in range(BUFFERED_FRAMES):   # TODO 측정: 드라이버가 쌓아 두는 프레임 수만큼 버림
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera read failed")
    return frame
source = CubevisionSource(grab, lambda f: pipe(f, params), mission="B", source="cubevision:cv.hsv_ranges")
```

- 이 모듈은 cubevision · numpy · OpenCV를 import하지 않는다. cubevision 결과를 속성 이름으로 읽으므로 `FrameResult` 객체, 그 `to_dict()` 결과(다른 프로세스에서 JSON으로 받은 것), 큐브 목록 어느 것이든 `scene_from_cubevision(...)`에 넣을 수 있다. 그래서 cubevision이 없는 기구학 노트북에서도 패키지가 import된다.
- 어댑터를 `backend/cubevision/` 안이 아니라 이 패키지 안에 둔 이유도 같다: 백엔드의 의존성이나 다른 파일을 건드리지 않고, 기구학 팀원이 대응 규칙을 같은 곳에서 읽을 수 있다.
- 필드 대응: `color → color`, `center → center_px`, `quad → corners_px`(순서 그대로), `angle → image_angle_deg`(`to_dict()`가 89.996을 90.0으로 반올림할 수 있어 [0, 90)으로 다시 접음), `area → area_px2`, `score` · `color_score` · `flags` 그대로, `FrameResult.notes → Scene.notes`.

## 9. 캘리브레이션 팀원에게 — `PixelToWorld` 만들기

자기 방식(예: 카메라 내부 파라미터 + `cv2.solvePnP`)이 있으면 3절의 계약만 지키면 된다. 없다면 템플릿을 채운다. 템플릿에는 **숫자가 하나도 없다**. 행렬 · 높이 · 판 윗면 z · 이름은 전부 필수 인자다.

```python
from vision_interface import HomographyPixelToWorld, fit_homography

# 권장: 큐브를 알려진 자리에 여러 개(4개 이상, 경기장 전체에 고르게) 놓고 비전의 center_px를 받는다.
pixels = [(u1, v1), (u2, v2), ...]          # 비전이 보고한 윗면 중심 (Cube.center_px)
world  = [(x1, y1), (x2, y2), ...]          # 그 큐브들 윗면 중심의 로봇 프레임 x, y (mm)
p2w = HomographyPixelToWorld(
    fit_homography(pixels, world),          # 또는 cv2.findHomography(...)로 구한 3x3
    calibrated_height_mm=25.0,              # 맞춘 평면 = 판 위 25 mm (큐브 윗면)
    board_top_z_mm=...,                     # 로봇 프레임에서 판 윗면의 z (C2)
    name="homography 경기장B 2026-11-29 18:20",
)
```

- 판 위 표시(z = 0)로 맞췄다면 `calibrated_height_mm=0.0`으로 두고, 다른 높이를 바꾸려면 `camera_xyz_mm=(x, y, z)`(로봇 프레임의 렌즈 중심)를 줘야 한다. 없으면 25 mm 높이 요청에 `ValueError`를 낸다(4.3절의 4.3 % 오차를 조용히 넣지 않기 위해). 렌즈 중심이 있으면 광선을 따라 정확히 옮긴다(왜곡 없는 핀홀 카메라, 기울어져 있어도 성립).
- 규정집의 렌즈 중심 CAD 값은 규정집 좌표계로 (−13.1, 131.4, 601.0)이다(§1.3 p.1, `rules.RULEBOOK_CAMERA_LENS_MM`). 이 패키지 코드는 이 값을 어디서도 쓰지 않는다. 실제 거치 상태와 맞는지는 캘리브레이션 팀원이 판단한다.
- `world_to_pixel(x, y, height_mm)`으로 로봇 목표점을 영상 위에 그려 확인할 수 있다.

## 10. JSON (다른 프로세스 · 다른 언어와 주고받기)

`scene.to_json()` / `Scene.from_json(text)` / `to_dict()` / `from_dict()`. 실수는 그대로 왕복한다.

```json
{
  "schema": "robothon.vision.scene", "schema_version": 1,
  "frame_id": 12, "timestamp_s": 1795780000.25, "image_size_px": [1280, 720],
  "mission": "B", "source": "cubevision:cv.hsv_ranges", "localized_by": null, "notes": [],
  "cubes": [
    {"color": "red", "center_px": [412.5, 300.25],
     "corners_px": [[380.1, 270.0], [444.0, 268.9], [445.2, 331.0], [381.0, 332.4]],
     "image_angle_deg": 1.2, "area_px2": 3901.5, "score": 0.97, "color_score": 0.88,
     "flags": ["side_band"], "pose": null}
  ]
}
```

`pose`가 있으면 `{"x_mm", "y_mm", "z_mm", "yaw_deg", "grasp_yaw_deg"}`. `schema_version`이 다르면 읽기를 거부한다(형식을 바꾸면 버전을 올린다). 모르는 키는 무시한다.

## 11. 규정에서 가져온 값

`rules.py`에 출처와 함께 있다. 규정집 = `Pick and Place 대회 규정집.pdf`(8쪽), OT = `26_로보톤_OT.pptx.pdf`.

| 값 | 코드 | 출처 |
|---|---|---|
| 좌표계: 원점 = 베이스 회전축, +x 오른쪽, +y 기둥 쪽, z = 0 판 윗면, 단위 mm | 4.2절, `rules.py` 머리말 | 규정집 §1.1 p.1, 도면 1/3 주기 1 (p.1) |
| 구역 반경 R90–R250, 외곽 R250.5 | `ZONE_R_MIN_MM`, `ZONE_R_MAX_MM` | §1.2 p.1, 도면 1/3 주기 3 |
| 렌즈 중심 (−13.1, 131.4, 601.0), 수직 아래 | `RULEBOOK_CAMERA_LENS_MM` (미사용) | §1.3 p.1, 도면 3/3 A (p.2) |
| 판 두께 17.5 mm | `BOARD_THICKNESS_MM` | §1.6 p.1, 도면 3/3 A |
| 큐브 25 mm 정육면체, 10 g, 나무 + R/G/B 인쇄 색종이 | `CUBE_SIZE_MM`, `CUBE_TOP_HEIGHT_MM` | §3 p.5, 도면 3/3 C (p.2) |
| 미션 A 큐브: R, G, B 각 3 (나무 없음) | `MISSION_CUBES[Mission.A]` | §3 p.5, §5 p.7, 도면 3/3 C |
| 미션 B 큐브: R, G, B, 나무 각 3 | `MISSION_CUBES[Mission.B]` | §4 p.6, §5 p.7 |
| 배치 순서 R → G → B → 나무, 1–20번 위치에서 무작위 추출 | `MISSION_CUBES` 순서, `PLACEMENT_SLOTS` | §5 p.7 |
| 미션 A 구역 1 = 0–60°·120–180°, 구역 2(탑) = 60–120° | `MISSION_A_ZONE1`, `MISSION_A_ZONE2` | §3 p.5, 도면 1/3 주기 4 |
| 미션 B 배치 구역 90–180°, 바구니 B 0–20° · G 20–50° · R 50–90° | `MISSION_B_PLACEMENT`, `MISSION_B_BASKETS` | §4 p.6, 도면 2/3 주기 2 · 5 (p.2) |
| 바구니 높이 35 (판 위 32), 벽 7, 안쪽 바닥은 판보다 4 높음 | `BASKET_RIM_HEIGHT_MM`, `BASKET_WALL_MM`, `BASKET_FLOOR_HEIGHT_MM` | §4 p.6, 도면 3/3 B |
| 바구니 배점 R 10 · G 15 · B 20 | `BASKET_POINTS` | §4 p.6 |
| 탑 최대 9층, 목표 색 = 1–3층 순서를 반복 | `TOWER_MAX_LEVELS`, `tower_target_colors` | §3 p.4–5, §5 p.7 |
| 층마다 25 mm씩 높게 놓음 | `tower_top_height_mm` | OT p.10 |
| 기둥 하단 필렛 R10 영역에는 블록 없음 | (가짜 소스에 미반영, 위치 치수 없음) | §1.4 p.1 |

규정집에 없는 것(만들어 넣지 않음): 카메라 기종 · 해상도 · 왜곡, 영상이 로봇 프레임에 대해 돌아간 방향, 로봇 베이스의 실제 z, 1–20번 위치 좌표, 기둥의 y 위치 치수, 큐브 방향.

**OT와 규정집이 다른 곳 (규정집을 따름)**

| 항목 | OT | 규정집 (따름) |
|---|---|---|
| 미션 A의 나무 블록 | "흩어진 블록(나무 포함)" (p.8) | R, G, B 각 3개만 (§3 p.5, 도면 3/3 C) |
| 나무 블록 감점 | "집거나 바구니에 넣으면 감점" (p.9) | 경기 끝에 바구니 안에 있는 블록만 판정, 색이 다르면(나무 포함) 감점 (§4 p.6) |
| 블록 배치 | "무작위로 흩뿌려 배치" (p.6) | 표시된 1–20번 위치에서 무작위 추출, R → G → B → 나무 순 (§5 p.7) |
| 시간 | 15:00–18:00 (p.5) | 18:00–20:45, 세팅 · 캘리브레이션 18:00–18:30 (타임라인 p.4) |
| 카메라 | 팀별 대여 (p.12), 운영진 거치대로 고정 (p.6) | 경기 중에는 운영진 카메라를 고정해 사용 (§1.5 p.1) |

## 12. 열린 질문 (정해야 할 것)

캘리브레이션 팀원과:

- [ ] **C1 로봇 프레임** — `PixelToWorld` 출력 프레임을 규정집 좌표계(§1.1 p.1: 원점 = 베이스 회전축, +x 오른쪽, +y 기둥 쪽, 오른손, z 위)로 할지. 다르면 `rules.py`의 구역 · 바구니 각도와 `pose.grasp_yaw_deg`의 기준(원점 = 회전축)이 맞지 않는다.
- [ ] **C2 z 기준** — 로봇 프레임에서 판 윗면의 z(`board_top_z_mm`). 판은 책상보다 17.5 mm 높고(§1.6 p.1) 베이스는 원점의 사다리꼴 홈에 들어간다(도면 1/3 p.1). 베이스가 책상에 놓이면 판 윗면은 베이스 바닥보다 17.5 mm 위일 수 있다 — 실측 필요.
- [ ] **C3 맞춘 평면** — 큐브 윗면(판 위 25 mm)에 맞출지(권장), 판 표시(z = 0)에 맞추고 렌즈 위치를 줄지(4.3절, §1.3 p.1).
- [ ] **C4 같은 프레임** — 경기 중엔 운영진 카메라를 쓴다(§1.5 p.1). 기종 · 해상도는 규정집에 없다(1280×720 C270은 우리 가정). 비전과 같은 해상도 · 같은 왜곡 처리(또는 무처리)의 프레임에서 맞출 것.
- [ ] **C5 다시 맞추는 때** — 세팅 · 캘리브레이션 18:00–18:30(타임라인 p.4), 시도마다 준비 2분에 로봇 설치 · 카메라 연결(§2 p.3). 로봇을 다시 놓을 때 빠르게 검증할 점 몇 개를 정해 둘지.

기구학 팀원과:

- [ ] **K1 그리퍼 yaw** — 손목 회전 관절의 영점 · 부호와 `pose.yaw_deg`(로봇 프레임 +x → +y)의 관계 (5절).
- [ ] **K2 언제 부르나** — `get_scene()`은 호출 뒤 새 프레임을 찍는다. 팔이 큐브를 가리지 않는 대기 자세를 정하고, `partial` 큐브는 다시 볼지.
- [ ] **K3 연결 방식** — 같은 프로세스에서 import할지, 별도 프로세스로 JSON(schema v1)을 주고받을지. 시작 · 종료 키(Space / q, §7 p.8)는 누구 루프가 받나.
- [ ] **K4 탑 색 순서** — 시도마다 추첨해 알려 준다(§5 p.7). 비전은 볼 수 없으니 누가 어떻게 입력할지(`rules.tower_target_colors`).
- [ ] **K5 높은 큐브** — 탑 꼭대기(25·n mm)와 바구니 안 큐브(29 mm)는 `localize_cube`로 다시 계산. 층수는 작업 상태에서.
- [ ] **K6 나무 블록(미션 B)** — 규정집상 집는 것 자체는 감점이 아니고 바구니 안에 있을 때만 감점(§4 p.6). 피할지 치울지.
- [ ] **K7 불확실한 검출** — `color == UNKNOWN`, 낮은 `score` · `color_score`를 어디서 거를지.
- [ ] **K8 리허설 위치** — 1–20번 위치가 당일 공개되면(§5 p.7) 좌표를 `positions_mm`로 넣어 가짜 소스로 리허설.

## 13. 테스트

```bash
cd ~/Desktop/로보톤/vision-lab/interface
python -m pytest -q                                   # pytest만 있으면 됨 (cubevision 연동 테스트 1개는 건너뜀)
../backend/.venv/bin/python -m pytest -q              # 백엔드 venv: 실제 cubevision 파이프라인까지 시험
```

| 파일 | 확인하는 것 |
|---|---|
| `tests/test_scene_json.py` | JSON 왕복(픽셀만 · 위치 포함), 스키마 버전 거부, 픽셀/pose 섞임 거부, 값 검사, JSON 파일 재생 |
| `tests/test_homography.py` | 알려진 배율 · 이동, 4점 정확 맞춤, 투영 행렬 복원, 높이 보정(손 계산), 렌즈 위치 없을 때 거부, 템플릿에 기본값 없음 |
| `tests/test_geometry.py` | `fold90`(−1e-17 포함), `closest_grasp_yaw` 표, 회전 · 뒤집힌 영상에서도 로봇 yaw가 맞음, localize 규칙, 필터 · 정렬, 탑 색 순서, `zone_of` |
| `tests/test_fake.py` | 기본 경로에 mm 없음, 시드 200개 × 미션 2개에서 수 · 색 · 순서 · 구역 · 겹침 · 영상 안, 1–20 위치 추출, `remove`, CLI, 6절 예제 실행 |
| `tests/test_cubevision_adapter.py` | cubevision `to_dict()` 모양의 결과 · 객체 · 목록, 각도 90.0 접기, `CubevisionSource` 순서, 실제 `cv.hsv_ranges` 파이프라인 결과(가능할 때) |
