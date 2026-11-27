# Vision Lab 설계

로보톤(SO-101 pick & place, 2026-11-29) 비전 담당이 여러 비전 모델을 이미지·영상으로 바로 시험하고,
결과를 크게 보고, 시험 기록을 쌓아 두기 위한 로컬 웹앱.

## 1. 목적과 범위

사용자 요구 (2026-11-25):

1. 다양한 비전 모델을 시험한다. 입력은 이미지와 영상.
2. 사이트 안에서 이미지·영상을 바로 올려 바로 시험한다.
3. 결과 이미지·영상을 크게 바로 본다.
4. 시험 기록을 관리한다.
5. 데이터를 사이트에 저장해 계속 이어서 시험한다.

사용자 결정 (같은 날 질문에 대한 답):

| 항목 | 결정 |
|---|---|
| 기본 탑재 모델 | OpenCV 고전 파이프라인만 (학습 불필요, 슬라이더로 즉시 재실행) |
| 학습형 모델 | 사용자가 학습한 Ultralytics `.pt`(detect / segment / obb)를 올려 시험. 학습 자체는 사이트 밖 |
| 채점 | 2026-11-27 범위에 넣음: 데이터셋에 정답셋(JSON)을 올려 일괄 실행을 자동 채점한다. 규칙은 `docs/SCORING.md`가 기준. 눈으로 비교 · 처리 속도 기록도 그대로 |
| 디스크 | 사용자가 정리함 (2026-11-25 기준 여유 43 GB) |

캘리브레이션(픽셀 → mm)은 다른 팀원 담당이라 범위 밖이다. 모든 출력은 이미지 픽셀 좌표다.

## 2. 용어

| 용어 | 뜻 |
|---|---|
| 미디어 (Media) | 올린 이미지 또는 영상 1개. 같은 파일(sha256)은 한 번만 저장 |
| 프록시 (Proxy) | 영상을 브라우저와 서버가 같은 프레임 번호로 읽도록 다시 인코딩한 H.264 CFR mp4. 원본은 그대로 보관 |
| 모델 (Model) | 프레임 → 큐브 목록. `cv.*` = 내장 OpenCV 파이프라인, `w.<id>` = 올린 가중치 |
| 파라미터 (Params) | 모델마다 정해진 조절값. 화면 컨트롤은 `ParamSpec`에서 자동 생성 |
| 프리셋 (Preset) | 이름 붙여 저장한 파라미터 묶음 |
| 미리보기 (Preview) | 슬라이더를 움직일 때마다 현재 프레임 1장에 파이프라인을 즉시 다시 돌린 결과. 기록에 남지 않음 |
| 실행 기록 (Run) | 미디어 1개 × 모델 1개 × 파라미터 1벌의 실행. 영상이면 여러 프레임. 기록에 남음 |
| 큐브 (Cube) | 결과의 단위. 색(red/green/blue/wood/unknown), 윗면 중심, 윗면 네 꼭짓점, 각도[0, 90), 점수 |
| 정답셋 (GT set) | 데이터셋에 올린 정답 JSON 1개(41장판, 410장판 등). 정답 이미지마다 큐브 라벨 |
| 연결 (Link) | 정답 이미지 ↔ 데이터셋 이미지. sha256이 같으면 자동(`sha256`), 아니면 사용자가 확인한 것만(`confirmed`) |

## 3. 구조

```
vision-lab/
├── backend/                 uv 프로젝트 (Python 3.12)
│   ├── cubevision/          순수 비전 패키지 — 웹 의존성 없음, 대회 코드에서 그대로 import 가능
│   │   ├── types.py         Cube · FrameResult · ParamSpec (계약)
│   │   ├── pipelines/       내장 OpenCV 파이프라인 (base.py 인터페이스)
│   │   ├── postprocess.py   마스크 · 폴리곤 · 박스 → 윗면 사각형 · 중심 · 각도 (모든 모델 공용)
│   │   ├── colour.py        색 분류
│   │   ├── roi.py           검출 영역(포함/제외 폴리곤)
│   │   ├── gt.py · score.py 정답셋 읽기 · 채점 계산 (docs/SCORING.md)
│   │   └── synth.py         합성 장면 생성기 (테스트 · 샘플용)
│   ├── lab/                 FastAPI 앱 (API · 저장 · 작업자 프로세스 · Ultralytics 어댑터 · 정답셋과 채점)
│   └── tests/
├── frontend/                React 19 + Vite 7 + TypeScript (빌드 결과를 FastAPI가 서빙)
├── data/                    (git 제외) lab.sqlite3, media/, proxy/, thumbs/, weights/, exports/, tmp/, gt/
└── docs/                    DESIGN.md, SCORING.md (채점 규칙), PIPELINES.md
```

프로세스:

- **API 프로세스** (uvicorn, 127.0.0.1:8010): REST API + 빌드된 프론트엔드. OpenCV 미리보기는 여기서 바로 계산(수십 ms). torch를 import하지 않는다.
- **작업자 프로세스** (API가 `spawn`으로 띄움): 실행 기록(Run)을 큐 순서대로 처리. Ultralytics 모델은 여기서만 로드(한 번에 1개). 취소는 프레임 사이에서 확인, 응답 없으면 강제 종료 후 재시작.
- 개발 중에는 Vite dev 서버 127.0.0.1:5190 이 `/api`를 8010으로 넘긴다.

## 4. 저장

SQLite (stdlib `sqlite3`, WAL, 외래키 on, `PRAGMA user_version`으로 마이그레이션):

```sql
media(id, sha256 UNIQUE, kind, name, ext, bytes, width, height, fps, frame_count, duration,
      tags JSON, note, starred, created_at)
weights(id, name, file, bytes, sha256, task, class_names JSON, imgsz, state, error, note, created_at)
presets(id, model_key, name, params JSON, created_at, UNIQUE(model_key, name))
runs(id, media_id → media ON DELETE CASCADE, model_key, model_name, params JSON,
     frame_start, frame_end, stride, frames_total, frames_done, status, cancel_requested,
     device, timing JSON, summary JSON, error, parent_run_id → runs ON DELETE SET NULL,
     tags JSON, note, starred, created_at, started_at, finished_at,
     batch_id → batches ON DELETE SET NULL)
run_frames(run_id → runs ON DELETE CASCADE, frame, ms, cubes JSON, PRIMARY KEY(run_id, frame))
datasets(id, name UNIQUE, note, created_at)
dataset_folders(dataset_id → datasets ON DELETE CASCADE, path, created_at, PRIMARY KEY(dataset_id, path))
dataset_media(dataset_id → datasets ON DELETE CASCADE, media_id → media ON DELETE CASCADE,
              folder, added_at, PRIMARY KEY(dataset_id, media_id))
batches(id, dataset_id → datasets ON DELETE SET NULL, dataset_name, folder, model_key, model_name,
        params JSON, stride, created_at)
gt_sets(id, dataset_id → datasets ON DELETE CASCADE, name, schema_version, label_kind, file_sha256, file_bytes,
        n_images, n_cubes, n_valid, created_at, UNIQUE(dataset_id, file_sha256))
gt_images(gt_set_id → gt_sets ON DELETE CASCADE, gt_index, sha256, filename, variant, layout_group,
          PRIMARY KEY(gt_set_id, gt_index))
gt_links((gt_set_id, gt_index) → gt_images ON DELETE CASCADE, media_id → media ON DELETE CASCADE, created_at,
         PRIMARY KEY(gt_set_id, gt_index), UNIQUE(gt_set_id, media_id))
```

데이터셋 안의 폴더 (마이그레이션 3):

- 폴더는 경로로 저장한다: 이름을 `/`로 이은 것(`증강/어둡게`). 부모 폴더도 늘 행이 있으므로 빈 폴더도 남는다.
- 미디어는 한 데이터셋에 한 번만 있고(기본 키 그대로) 그 데이터셋의 폴더 하나에 있다(`dataset_media.folder`, `''` = 맨 위).
- 경로 규칙(서버에서 검사, 어기면 400): NFC로 맞추고 이름마다 앞뒤 공백을 지운다. `.` `..` 빈 이름(`//`, 끝의 `/`),
  `/`나 드라이브 문자로 시작하는 절대 경로, `\`, 제어 문자는 받지 않는다. 이름은 80자, 깊이는 8단계까지.
  같은 부모 아래 이름은 대소문자만 다른 것도 겹칠 수 없다(압축을 풀면 macOS·Windows에서 한 폴더가 된다).
  올리기·옮기기에서 대소문자만 다른 경로는 있는 폴더로 맞춘다.
- 폴더 이름 바꾸기 = 그 경로로 시작하는 폴더·미디어·그 폴더 일괄 실행의 `batches.folder`를 함께 바꾼다(다른 곳으로 옮기기도 같다).
- 폴더 지우기 = 하위 폴더까지 지우고 그 미디어를 **이 데이터셋에서만** 뺀다. 미디어 파일·실행 기록·일괄 실행은 남는다(미디어 빼기와 같다).
- 일괄 실행은 데이터셋 전체(`batches.folder` NULL) 또는 폴더 하나와 그 아래 전부.
- zip은 폴더를 디렉터리로 쓴다: 원본 zip `<폴더>/<파일>`, 결과 zip `<폴더>/<이름>_result.<확장자>`. 맨 위 미디어는 zip 맨 위.
  이름이 겹치면 폴더마다 `-2`, `-3`을 붙인다. 폴더 일괄 실행의 결과 zip도 경로는 데이터셋 기준이라 여러 폴더의 zip을 한곳에 풀어도
  그대로 합쳐진다. results.json 항목과 results.csv에 `folder`(zip을 만들 때 그 미디어가 있는 폴더)가 있다.

정답셋 (마이그레이션 4, 규칙은 `docs/SCORING.md` §1):

- 정답셋은 데이터셋에 속한다. 올린 JSON은 바이트 그대로 `data/gt/gt-<id>.json`에 두고, 이미지 목록(sha256 · 파일 이름 · 증강 ·
  layout_group)을 `gt_images`에 색인한다. 같은 데이터셋에 같은 파일(sha256)을 다시 올리면 409, 다른 데이터셋에는 따로 올릴 수 있다.
- sha256 연결은 저장하지 않는다. 그때그때 데이터셋의 현재 이미지에서 구하므로 나중에 넣은 미디어도 저절로 연결된다.
  `gt_links`에는 사용자가 확인한 연결만 있다. 확인한 연결은 그 미디어가 데이터셋의 이미지이고, 그 정답 이미지나 그 미디어를
  sha256 연결이 차지하지 않을 때만 쓰인다(sha256 우선). 나머지는 무시하고 이유와 함께 알린다(`links.ignored`).
- 데이터셋을 지우면 정답셋 행과 파일도 지운다. 정답셋을 지워도 미디어 · 실행 기록 · 일괄 실행은 남는다.

파일: `data/media/<sha256>.<ext>` 원본, `data/proxy/<sha256>.mp4`, `data/thumbs/media-<id>.jpg`,
`data/thumbs/run-<id>.jpg`, `data/weights/<id>.pt`, `data/exports/run-<id>.(png|mp4)`, `data/gt/gt-<id>.json` (정답셋 원본).
Ultralytics 설정·캐시도 `data/ultralytics/`로 돌린다(자동 설치·통계 전송 끔).

실행 결과는 큐브 JSON만 저장하고 결과 이미지·영상은 브라우저에서 원본 위에 다시 그린다.
구운(baked) PNG/MP4는 사용자가 "내보내기"를 누를 때만 만든다.

## 5. API (`/api`)

| 메서드 · 경로 | 내용 |
|---|---|
| `GET /health` | 살아 있는지 → {ok, build(프런트 빌드가 바뀌면 바뀜), features(이 서버에 있는 경로 묶음: `folders` · `gt` · `score`. 화면은 404 문장 대신 이것으로 기능 유무를 판단)} |
| `GET /system` | 버전, torch/MPS 사용 가능 여부, 작업자 상태, 디스크 |
| `GET /media` `?kind&q&starred&tag` | 미디어 목록 (최신순) |
| `POST /media` (multipart `file`) | 올리기 → `{media, duplicate}`. 영상은 프록시를 만든 뒤 응답 |
| `GET/PATCH/DELETE /media/{id}` | 조회 · 이름/태그/메모/별표 수정 · 삭제(실행 기록과 파일까지) |
| `GET /media/{id}/file` | 이미지 원본 또는 영상 프록시 (Range 지원) |
| `GET /media/{id}/original` | 원본 다운로드 |
| `GET /media/{id}/thumb` | 320 px 썸네일 |
| `GET /media/{id}/frame/{n}` | 프레임 n JPEG |
| `GET /models` | 내장 파이프라인 + 올린 가중치 |
| `POST /models/weights` (multipart `file`, `name`) | `.pt` 올리기 → 작업 종류·클래스 이름을 읽어 등록 |
| `PATCH/DELETE /models/{key}` | 가중치 이름/메모 수정 · 삭제 |
| `GET /presets?model_key` · `POST /presets` · `DELETE /presets/{id}` | 프리셋 |
| `POST /params/check` {model_key, params} | 손으로 쓴(ChatGPT가 준) 파라미터를 실행과 같은 규칙으로 맞춤 → params(전체), applied, adjusted, ignored |
| `GET/POST /datasets` · `PATCH/DELETE /datasets/{id}` · `POST /datasets/{id}/media` · `POST /datasets/{id}/media/remove` | 데이터셋(미디어 묶음, 다대다). 지워도 미디어·기록은 남음. 응답의 `folders` = 폴더마다 path · count(바로 안) · total(하위 포함). `GET /media?dataset_id=&folder=`(그 폴더와 하위 폴더, 미디어마다 `folder`), `POST /media?dataset_id=` + 폼 필드 `folder`(올리면서 그 폴더에 넣기, 없으면 만듦. 이미 데이터셋에 있던 파일은 제자리에 두고 `already_in_dataset`) |
| `POST /datasets/{id}/folders` {path} · `POST /datasets/{id}/folders/rename` {path, new_path} · `POST /datasets/{id}/folders/delete` {path} · `POST /datasets/{id}/media/move` {media_ids, folder} | 폴더 만들기(없는 부모도) · 이름 바꾸기/옮기기(하위 폴더·미디어도 함께) · 지우기(하위 폴더까지, 미디어는 이 데이터셋에서만 빠짐) · 미디어를 다른 폴더로(`''` = 맨 위). 응답은 데이터셋 |
| `GET /datasets/{id}/originals.zip` | 데이터셋 원본 전부(폴더 구조 그대로) |
| `GET/POST /datasets/{id}/batches` · `GET /batches/{id}` · `POST /batches/{id}/cancel` · `DELETE /batches/{id}` | 일괄 실행: 데이터셋(또는 `folder`로 준 폴더와 그 아래)의 미디어마다 같은 모델·파라미터로 실행 1개(runs.batch_id). 일괄 실행과 항목에 `folder`. 삭제하면 그 실행 기록도 삭제 |
| `GET /batches/{id}/results.zip` · `GET /runs/{id}/download` | 끝난 실행의 그려진 결과(PNG/MP4) + results.json + results.csv / 실행 하나의 그려진 결과 |
| `POST /preview` | 파이프라인 1프레임 즉시 실행 → 큐브 + 단계별 시간 + 디버그 레이어(PNG) |
| `GET /runs` `?media_id&model_key&status&starred&tag&limit&offset` | 실행 기록 목록 |
| `POST /runs` | 실행 기록 만들기(큐에 넣음) |
| `GET/PATCH/DELETE /runs/{id}` | 조회 · 메모/태그/별표 · 삭제 |
| `POST /runs/{id}/cancel` | 취소 |
| `GET /runs/{id}/frames` | 프레임별 큐브 |
| `GET /runs/{id}/thumb` | 결과가 그려진 썸네일 |
| `POST /runs/{id}/export` · `GET /runs/{id}/export` | 결과를 구운 PNG/MP4 만들기 · 받기 |
| `POST /datasets/{id}/gt` (multipart `file`, `name`) | 정답셋 올리기(최대 50 MB, 이름 기본값 = 파일 이름) → GtSetInfo(개수 · 증강 · 연결 보고). 형식 오류 400(파서의 문장), 같은 파일 409 |
| `GET /datasets/{id}/gt` | 데이터셋의 정답셋(최신순). 연결 수: 정답 이미지, sha256 연결, 확인 연결, 연결 안 됨, 정답 없는 데이터셋 이미지, 영상(채점 안 함), 무시된 확인 연결 |
| `GET/PATCH/DELETE /gt/{gid}` | 상세(연결된 이미지 · 연결 안 된 정답 이미지와 후보(파일 이름이 같은(NFC) 미연결 이미지) · 정답 없는 이미지) · 이름 바꾸기 · 지우기(파일까지) |
| `POST /gt/{gid}/links` {links: [{gt_index, media_id}]} · `DELETE /gt/{gid}/links/{gt_index}` | 연결 확인(전부 아니면 전무: 데이터셋의 이미지일 것, 정답 이미지가 미연결일 것, 미디어가 이 정답셋의 다른 정답 이미지에 연결되지 않았을 것) → GtSetDetail · 확인 연결 끊기(sha256 연결은 400) |
| `POST /gt/{gid}/organize` {root} | 연결된 미디어를 `<root>/<증강 이름>` 폴더로 옮김(폴더가 없으면 만듦) → Dataset |
| `GET /media/{id}/gt` | 이 미디어를 연결한 정답셋(모든 데이터셋, 최신순)과 gt_index · 연결 방식 |
| `POST /score` {gt_set_id, batch_ids, criteria, reference_batch_id, detail} | 일괄 실행 채점(`batch_ids` 비우면 그 데이터셋의 전부, 최신순). 배치마다 상태(complete / interim), `pending`(끝나지 않은 이유별 수와 `total`), 평가 이미지 수 `n_eval_images`(포함률 = 이것 / `gt_set.n_images`) · `full_coverage`, 지표(채점된 이미지 수 = `metrics.n_images`, 실패 실행 = `metrics.n_failed`) · 조각 · 증강 영향, 기준 배치와의 paired bootstrap(또는 비교하지 않은 이유) + `final_ranking`(완료 · 전체 포함 배치의 id, 허용 오차 충족률 → 이미지 전체 기준 충족률 순. 차이 판정은 각 배치의 `comparison`). `detail: "summary"`(기본 `"full"`)는 상태 · pending · 포함률 · 지표만 내고 `slices` · `augmentation_effect` · `comparison`은 null(bootstrap 안 함, 왼쪽 띠의 점수 칩용). 수는 모두 n / 분자 / 분모와 함께, 없는 값은 null |
| `GET /score/criteria` | 기준값의 기본값과 한계 → {defaults, limits: {이름: {min, max, min_exclusive, max_exclusive, step, integer}}}. 요청 스키마(`ScoreCriteriaIn`, OpenAPI에도 그대로)에서 읽으므로 기본값 · 한계는 한 곳(= `Criteria` · `MAX_BOOTSTRAP_REPS`)에만 있다. `step`은 입력 화살표용 힌트. 기준값 필드는 빠지거나 null이면 기본값 |
| `GET /score/image?run_id&gt_set_id&<기준값>` · `POST /score/frame` {media_id, cubes, gt_set_id, criteria} | 이미지 하나의 채점 상세: 정답셋 이름 · gt_index · 연결 방식, 정답 큐브(id · 색 · 기하 유효 · quad · center · match_point), 검출(큐브 그대로), 정답 큐브마다 결과(짝 `pred_index` · `distance`, 색, 세 오차, 기준별 통과 `center_ok` · `angle_ok` · `vertex_ok` · `geom_ok`, `passed`, `outcome` = pass / fail / missed), 오검출 번호, 이미지 통과, `/score/images`와 같은 수(정답 · 미검출 · 오검출 · 색 틀림 · 기하 실패), 쓴 기준값. `gt_set_id` 없으면 그 미디어를 연결한 가장 최근 정답셋. 정답이 없으면 404 |
| `GET /score/images?gt_set_id&batch_id&<기준값>` | 배치 하나의 평가 이미지 전부(정답 순서) → ScoreImageRow[]: gt_index, 미디어 id · 이름 · 폴더, 정답 파일 이름, run_id(배치에 실행이 없으면 null), 증강, source_sha256, layout_group, 상태(실행 상태 또는 `missing`), 이미지 통과, 정답 수, 미검출 · 오검출 · 색 틀림 · 기하 실패 수(기하 유효 짝이 τ를 넘거나 중심 · 각도 · 꼭짓점이 없음), ms. 끝나지 않은 이미지는 수가 null. 수는 `/score/image`와 같은 함수(`cubevision.score.outcome_counts`)로 세고, `POST /score`와 같은 이미지 채점 캐시를 쓴다(채점 화면의 원본별 · 이미지별 결과) |
| `GET /score/cubes.csv` · `GET /score/summary.csv` (`?gt_set_id&batch_ids=1,2&<기준값>`) | `docs/SCORING.md` §7의 두 CSV(BOM, 기준값 열). 중간 배치도 `status` 열(완료 / 중간)과 함께 |
| `GET /storage` · `POST /storage/cleanup` | 용량 · 정리 |
| `POST /worker/restart` | 작업자 재시작 |
| `POST /samples` | 합성 샘플 이미지·영상 넣기 (태그 `sample`) |

정확한 요청·응답 형식은 `backend/lab/schemas.py` 와 `frontend/src/api/types.ts` 가 기준이다.
오류는 `{detail: "<한국어 문장>"}`.

## 6. 화면

1. **작업대 `/`** — 왼쪽 미디어 목록(끌어다 놓기 · 붙여넣기 · 파일 선택으로 올리기), 가운데 큰 뷰어,
   오른쪽 모델 선택 + 파라미터 컨트롤 + 결과 큐브 표 + 처리 시간, 아래쪽 이 미디어의 실행 기록 띠.
   파이프라인은 슬라이더를 움직이면 바로 미리보기가 갱신되고, "기록 저장"을 누르면 실행 기록이 된다.
   가중치 모델은 "실행"을 누르면 작업자가 처리한다.
   - **데이터셋과 폴더** (왼쪽 위): 데이터셋을 고르면 목록과 올리기가 그 데이터셋으로 좁혀지고, 그 아래 폴더 나무
     (펼치기/접기, 폴더마다 하위 포함 개수)가 나온다. 폴더를 고르면 목록은 그 폴더와 하위 폴더(아래 폴더의 항목은
     이름 앞에 상대 경로), 올리기는 그 폴더로 간다. "새 폴더"는 고른 폴더 안에 만든다. 고른 폴더의 연필 = 이름 바꾸기,
     휴지통 두 번 = 지우기. "선택"으로 고른 미디어를 다른 폴더로 옮긴다. 일괄 실행 버튼은 고른 폴더(하위 포함) 또는
     전체를 실행하고, 일괄 실행 목록에 범위("전체" 또는 폴더 경로)가 붙는다.
   - **폴더째 올리기** (폴더 버튼 또는 폴더를 끌어다 놓기, 하위 폴더까지 읽음): 파일마다 올린 폴더 안의 경로가 고른 폴더
     아래 폴더가 된다. 여러 개를 놓으면 각 폴더 이름이 그대로 남는다. 폴더 하나만 올렸는데 그 안에 폴더만 있거나
     (예: `원본/ 증강1/`을 담은 상위 폴더) 이름이 고른 폴더와 같으면, 그 폴더는 한 단계를 더 만들지 않고 내용만 들어간다.
     `.DS_Store`, `Thumbs.db`, `._`로 시작하는 파일은 건너뛴다. 데이터셋에 이미 있던 파일은 제자리에 두고
     "이미 데이터셋의 <폴더>에 있음, n개"로 알린다. 데이터셋을 고르지 않았으면 폴더 구조 없이 올린다.
2. **뷰어** (이미지 · 영상 공용) — 커서 기준 확대/이동, 맞춤 · 100 %, 전체 화면, 오버레이 켜고 끄기
   (윗면 사각형, 중심, 각도, 라벨, 원 출력, 디버그 마스크), 픽셀 정보(x, y, RGB, HSV, Lab),
   색 고르기(파라미터 `color`), 검출 영역(ROI) 그리기, 영상은 프레임 단위 이동과 결과 동기 표시.
3. **기록 `/runs`** — 썸네일 격자/표, 필터(미디어 · 모델 · 상태 · 별표 · 태그), 메모 · 태그 · 별표,
   삭제, "파라미터 바꿔 다시 실행", 두 기록 비교.
4. **비교 `/compare?a=&b=`** — 같은 미디어의 두 기록을 나란히(확대 · 재생 위치 연동) 또는 겹쳐 보기,
   큐브 수 · 색 · 시간 차이.
5. **모델 `/models`** — 내장 파이프라인 설명, `.pt` 올리기 · 목록 · 삭제, 저장 공간과 정리.
6. **채점 `/score?dataset=&gt=&ref=&skip=`** (`docs/SCORING.md`) — 위 막대: 데이터셋 · 정답셋 · 기준 배치, 기준값(τc · τa · τv ·
   g, bootstrap 반복 · 수준 · 시드, "잠정 기준 — 로봇 실제 허용 오차 아님" 늘 표시), 일괄 실행 체크(기본 전부), 정답셋이 AI 보조 근사 라벨이라
   차이는 paired bootstrap 구간으로만 판정한다는 안내, cubes.csv · summary.csv. 아래: 최종 순위(완료 · 전체 포함 배치만, 축별 열, 값마다
   분자/분모 또는 n, 기준 대비 Δ · 구간 · 높음/낮음/차이 불분명), 중간 결과 · 비교 제외(이유를 말로, 값은 중간 점수), 조각 비교(지표 고르기,
   큐브 단위 조각은 큐브 지표만), 증강 영향(증강마다 Δ와 판정), 배치 하나의 원본별 결과(원본 × 증강, 통과 · 미검출 · 오검출)와 이미지별
   결과(실패 많은 순, 누르면 작업대에서 정답 레이어를 켜고 그 실행을 연다). 처음 채점 · τ · g 변경은 몇 초 걸리므로 이전 결과를 흐리게 둔 채
   다시 채점하고, 기준값 입력은 0.4 s 멈춘 뒤 보낸다.
   - 기준값은 화면마다 따로가 아니라 하나다(브라우저 localStorage `vl1.score.criteria`): 채점 화면, 왼쪽 일괄 실행의 점수 칩, 뷰어의 정답 레이어,
     작업대의 실시간 채점이 모두 같은 값을 쓴다.
   - 작업대 데이터셋 칸의 **정답셋 줄**: 정답 JSON 올리기, 정답셋 고르기 · 이름 바꾸기 · 지우기, 연결 수("연결 409/410" 또는 "연결 안 됨 1"),
     연결 확인 창(연결 안 된 정답 이미지와 파일 이름이 같은 후보 — 확인해야만 연결, 줄마다 또는 "후보 모두 연결", 확인한 연결 끊기),
     두 번 눌러 "증강 종류별 폴더로 정리". 일괄 실행마다 점수 칩(허용 오차 충족률 · 이미지 충족률 · 완료/중간).
   - **뷰어 정답 레이어**(레이어 메뉴 "정답", 정답이 있는 이미지에서만): 정답 윗면 점선, 검출 중심 → 정답 매칭점 선, 미검출 · 오검출 표시,
     큐브에 올리면 짝 거리 · 중심 · 회전각 · 꼭짓점 RMS · 통과. 실행 기록은 `GET /score/image`, 미리보기는 실시간 채점 결과를 그린다.
   - **실시간 채점**(작업대 결과 칸, 정답이 있는 이미지에서만): 미리보기 · 실행 결과의 큐브를 `POST /score/frame`으로 채점해 이미지 통과,
     미검출 · 오검출 · 색 · 기하 수, 검출마다 통과/미달, "정답" 탭에서 정답 큐브마다 오차. 파라미터를 바꾸면 따라 바뀐다.
   - 정답셋 · 채점 경로가 없는 예전 서버(404)에서는 이 부분들을 숨기고 오류 알림을 띄우지 않는다. 채점 화면은 서버를 다시 켜라고 말한다.

## 7. 영상

- 올릴 때 `ffprobe`로 읽고 `ffmpeg`로 프록시를 만든다: H.264 yuv420p, 고정 프레임률, 회전 반영,
  B-프레임 없음, 키프레임 간격 15, `+faststart`. 서버 추론과 브라우저 재생이 같은 프록시를 읽으므로
  프레임 n은 양쪽에서 같은 그림이다. 브라우저 쪽 프레임 번호 = `round(mediaTime × fps)`
  (`requestVideoFrameCallback`).
- 실행 기록은 `frame_start`(포함)부터 `frame_end`(제외)까지 `stride`마다 처리한다.
  재생 중에는 현재 프레임 이하에서 가장 가까운 처리 프레임의 결과를 그린다.
- 내보내기 MP4는 프레임에 결과를 그려 ffmpeg(libx264, yuv420p, faststart)로 인코딩한다.
  (`cv2.VideoWriter`의 `mp4v`는 브라우저에서 재생되지 않는다.)

## 8. 모델

- **내장 OpenCV 파이프라인** (`cubevision.pipelines`): 서로 다른 발상의 파이프라인 여러 개.
  모두 같은 후처리(`postprocess.py`)로 윗면 사각형 · 중심 · 각도를 만든다. 목록은 9절.
- **올린 가중치** (`lab` 의 Ultralytics 어댑터): detect → 박스 안에서 윗면을 맞춤,
  segment → 마스크 폴리곤에서 윗면을 맞춤, obb → 회전 박스를 그대로 사각형으로.
  색은 클래스 이름이 색 이름이면 그것을, 아니면 영역 픽셀로 분류한다.
  파라미터: Ultralytics 예측 인자 중 결과·속도를 바꾸는 것 전부(conf, iou, imgsz, rect, max_det, classes, agnostic_nms, augment,
  nms, tile, batch, device, quantize, compile, channels_last, segment는 retina_masks) + 랩의 색·윗면 후처리 파라미터 + 검출 영역.
  목록과 뺀 것은 `docs/PIPELINES.md` 2절 '파라미터는 전부 공개한다'.

## 9. 내장 파이프라인 목록

표시 순서 = `cubevision/pipelines/__init__.py`의 `ALL`. 키는 실행 기록에 저장되므로 바꾸지 않는다.
발상·조절값·대회 날 맞추는 순서·합성 기준 측정표는 `docs/PIPELINES.md`.

| 키 | 이름 | 한 줄 |
|---|---|---|
| `cv.hsv_ranges` | HSV 색 범위 | 색마다 H·S·V 범위로 마스크를 만들고 그 범위가 곧 색인 가장 빠른 기본 방식 |
| `cv.board_diff_lab` | 판 차이 + Lab 최근접색 | 흰 판과 색(채도)이 다른 픽셀을 전경으로 보고, 픽셀마다 가장 가까운 기준색을 붙이는 방식 |
| `cv.edge_quads` | 윤곽으로 닫힌 사각형 | 색을 보지 않고 윤곽선이 둘러싼 윗면 크기의 사각형 영역을 찾은 뒤 색을 읽는 방식 |
| `cv.lab_autocluster` | Lab 자동 군집 | 판을 뺀 픽셀을 k-평균으로 묶고 각 군집에 가장 가까운 기준색 이름을 붙이는, 문턱 없는 방식 |

모두 같은 후처리(`postprocess.cubes_from_blobs`: 크기·모양 거르기, 붙은 큐브 나누기, 옆면 띠를 뺀 윗면, 판 기준 색 판정)를 쓰고,
모두 `roi`(검출 영역), `face_px`(윗면 한 변, 기본 61 px), 기준색 `proto_red/green/blue/wood`와 `proto_board`를 파라미터로 가진다.

## 10. 시험

- `cubevision`: 합성 장면(흰 판, 회전 · 원근 옆면 · 그림자 · 잡음 · 바구니 색 바닥)으로
  파이프라인마다 검출 수 · 색 · 중심 오차 · 각도 오차를 확인하는 pytest.
- `lab`: FastAPI TestClient로 올리기 → 미리보기 → 실행 → 목록 → 삭제, 합성 영상 실행.
  데이터셋 폴더: 경로 규칙, 폴더 만들기 · 이름 바꾸기(하위까지) · 지우기(미디어는 데이터셋에서만 빠짐), 폴더로 올리기와
  이미 있던 파일 알림, 폴더 일괄 실행, 두 zip의 정확한 경로와 results.json · results.csv의 `folder` (`test_lab_folders.py`).
  정답셋과 채점: 실제 두 정답셋 올리기(개수 · 409 · 400), sha256 자동 연결(나중에 넣은 미디어 포함), 후보는 확인해야 연결,
  전부 아니면 전무, sha256 우선, 변형 폴더로 정리, 손으로 계산한 지표(완벽 · 어긋남 · 실패 · 중간 · 폴더 배치), paired bootstrap,
  null 유지, 요약 응답(`detail: "summary"`), 이미지 상세와 목록의 같은 수, CSV, 기준값 기본값 · 한계, 캐시(폴링은 바뀐 실행만,
  정답셋마다 기준값 3세대), 데이터셋 삭제 (`test_lab_gt.py`). 캐시 자체(LRU, stamp, 동시 요청은 한 번만 계산):
  `test_lab_cache.py`.
- 화면: Playwright로 올리기 → 슬라이더 → 저장 → 기록 → 비교, 영상 오버레이가 맞는 프레임에 그려지는지.
  폴더: 폴더 만들기 → 폴더째 올리기 → 개수 → 한 폴더만 일괄 실행 → zip 경로 → 옮기기 · 이름 바꾸기 · 지우기,
  끌어다 놓은 폴더를 하위까지 읽는지 (`folders.spec.ts`).
  채점: 정답 JSON 올리기 → 연결 수 → 이름만 같은 파일은 확인 전엔 후보 → 확인 · 끊기 · 후보 모두 연결 → 폴더 정리, 일괄 실행 → 점수 칩 →
  채점 화면 최종 순위의 값(`make_media.py`가 파이프라인 출력에 알려진 변화 6개를 넣어 만든 정답셋과 손으로 낸 기대값), 대기 · 취소된 배치는
  중간 결과에만, τc를 바꾸면 값이 바뀜(다른 화면도 같은 기준), 두 CSV의 머리줄, 뷰어 정답 레이어, 실시간 채점, 경로가 없는 예전 서버
  (`scoring.spec.ts`). 백엔드: 배치의 이미지별 목록(`test_lab_gt.py`).

## 11. 범위 밖

캘리브레이션(픽셀 → mm), 사이트 안 학습, 제로샷 · SAM 계열 모델,
여러 사용자 · 로그인, 외부 공개.
