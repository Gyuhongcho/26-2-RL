# 연세대 연계 마을버스 노선 설계

기본 입력 `network_role4.json`은 팀원 자료의 공식 ARS 목록 464곳에서 학습 기간 하차 기록이 있고 중앙차로 제외 기준을 통과한 **412곳**을 사용한다. 팀의 300m 거점 매핑과 고정 척도 수요 점수를 적용한다. 9–10월 평가에서 방문 정류장 중 하차 기록이 비어 있으면 해당 월 평가는 null이다. 거점 보너스 기본값 2점과 거리 비용 0.2점/km는 기존 실험 설정이다.

입력 출처는 팀원 저장소 커밋 `48d4a845f215d1a4e915b1f671c621d3f1a45d80`의 `data/processed/role4_bundle.json`이다. 새 번들이 있으면 `python3 adapt_role4.py /path/to/role4_bundle.json --out network_role4.json`으로 재생성한다. `--diagnostic`을 붙여 생성한 `network_role4_diagnostic.json`은 원본 거리 행렬도 확보한 259곳이다.
배경 지도는 OpenStreetMap이다.

## 실행

```sh
python3 train.py
```

로컬 OSRM 서버를 먼저 시작한다. 기본 주소는 `http://127.0.0.1:5000`이다. 기본 실행은 Random → 이용량 대비 거리 Greedy → DQN → Monte Carlo Control 순으로 각각 1,000회 노선을 생성한다. 다섯 번째 화면에서 각 방법이 발견한 최고 보상 노선을 지도 네 장으로 비교한다. 출발점 순서, 거리 제약, 보상식은 공통이며 DQN과 Monte Carlo는 ε-greedy 탐색을 사용한다. Greedy는 같은 출발점에서 같은 경로를 생성한다. Monte Carlo는 노선이 끝날 때 실제 할인 누적보상의 표본평균으로 Q값을 갱신한다. 기본 비교를 실행하기 전에 `python3 -m pip install -r requirements.txt`로 PyTorch를 설치한다.

기록은 기본적으로 `runs/local`의 방법별 하위 폴더에 저장된다. 화면 없이 비교하려면 `python3 train.py --no-viewer --out runs/comparison`을 실행하고, 나중에 `python3 train.py --replay runs/comparison`으로 비교 지도를 연다.

### 방법 하나만 실행

Python 3.9 이상에서 Q-learning, SARSA, 기준 전략을 바로 실행할 수 있다.
DQN은 PyTorch가 필요하다. GPU 머신에는 CUDA를 지원하는 PyTorch를 설치한다.

```sh
python3 train.py --algo q-learning --episodes 5 --rounds-per-episode 20
```

VS Code에서 **Run Python File**을 누르면 기본 다섯 화면 비교가 실행된다. 실행하면 기본 브라우저가 열리고 버스 아이콘이 학습 중 생성한 경로를 순서대로 재생한다. 브라우저가 자동으로 열리지 않으면 터미널에 출력된 주소를 연다. 학습 종료 후에도 지도를 볼 수 있으며, 터미널에서 Ctrl+C를 누르면 종료한다. 지도 타일과 Leaflet에는 인터넷 연결이 필요하다. OSRM 모드에서는 버스가 저장한 도로 구간을 따라 움직인다. 도로 좌표가 없는 기록은 정류장 방문 순서를 직선 점선과 버스 아이콘으로 표시한다. 거리와 보상은 도로거리 기준이다.

```sh
python3 train.py --algo sarsa --steps 3000 --out runs/sarsa
python3 -m pip install -r requirements.txt
python3 train.py --algo dqn --steps 5000 --device cpu --out runs/dqn
python3 train.py --algo random --episodes 5 --rounds-per-episode 20 --out runs/random
python3 train.py --algo greedy --episodes 1 --rounds-per-episode 1 --out runs/greedy
python3 train.py --algo dp --max-stops 12 --episodes 1 --rounds-per-episode 1 --out runs/dp
```

지도는 현재 실행의 출력 폴더를 자동으로 읽는다. 실험별로 별도의 `--out`을 지정한다. GPU 서버나 일괄 실험에서는 `--no-viewer`를 붙이면 학습이 끝난 뒤 바로 종료된다.
대규모 비교 시 세 RL 방식에 각각 동일한 `--steps 200000`, 설정, 시드 목록을 적용한다. CUDA에서는 `--device cuda`를 사용한다.
DP 최적해와 비교할 때는 모든 방식에 같은 `--max-stops`를 지정한다. 후보 제한은 후보표 앞부분을 사용한다.

## 환경과 보상

- 상태는 현재 정류장, 방문 집합, 남은 거리다. 다음 정류장을 선택하면 방향별 도로거리를 차감한다.
- 거리를 더하면 상한을 넘는 정류장과 이미 방문한 정류장을 제외한다. 가능한 이동이 없으면 현재 정류장에서 종료한다.
- 기본 수요 점수는 팀 번들의 5–8월 오전 7–9시 하차량 기간 일평균과 고정 정규화 점수를 사용한다. 9–10월 값은 학습된 노선의 월별 경유지 하차량 평가에 사용한다.
- 점수 = 경유 정류장별 정규화 하차량 + 최초 거점 방문당 2점 − 도로거리 1km당 0.2점. 출발점 하차 점수는 제외하고 마지막 정류장 하차 점수는 포함한다.
- 이동 가능한 일반 정류장 중 `--start-seed`로 5곳을 한 번 뽑아 `starts.json`에 시드와 ID를 저장한다. 같은 후보와 거리 상한에서 같은 출발 시드를 사용하면 재현된다. `--seed`는 학습 난수용이다.
- 한 라운드에 저장한 출발점 5곳 중 하나를 뽑아 경로 1개를 생성한다. 기본 20라운드가 한 에피소드이며 기본 5에피소드에서 학습을 종료한다. 종점은 마지막으로 선택한 정류장이다. 기본 거리 상한은 10,000m이며 `--budget-m`으로 설정한다.
- 연대앞과 팀 자료의 정류장 이름에 역이 명시된 철도 거점 9곳에 최초 방문 보너스를 준다. 같은 역의 여러 출구는 하나로 묶고, 신촌역 2호선과 경의중앙선 역은 별개 거점으로 처리한다. 역 이름 대응표는 `prepare.py`에서 관리한다.
- 하차량은 기존 이용 실적이다. 거리 자료는 실제 마을버스의 회전 및 정차 가능성까지 검증한 값으로 해석하지 않는다.

## 비교 방식과 출력

Q-learning과 SARSA는 동일한 상태 표현으로 각각 다음 상태의 최대 Q와 다음에 실제 선택한 행동의 Q를 사용한다. DQN은 128–128 MLP, 경험 재생, 타깃 네트워크와 유효 행동 마스크를 사용한다. 세 방법 모두 `--exploration epsilon-greedy` 또는 `--exploration greedy`를 선택한다. 기준 탐욕법은 이용량 점수와 신규 거점 보너스의 합을 이번 이동거리로 나눈다. DP는 작은 후보 집합에서 벨만 최적값을 메모이제이션한다.

`history.jsonl`에는 모든 경로를, `episode_winners.json`에는 에피소드별 최고 보상 경로를 저장한다. 지도에서 **최고 보상 노선 보기**를 누르면 최신 에피소드 우승 경로를 재생한다. 모델은 `model.pt` 또는 `q_table.json`에 저장된다.
`--steps`를 지정하면 전이 예산에 맞춰 학습하고 마지막으로 진행 중이던 노선을 완성한다. 마지막 에피소드는 설정한 라운드 수보다 짧을 수 있다.

병렬 실험 명령은 [PARALLEL_EXPERIMENTS.txt](PARALLEL_EXPERIMENTS.txt)에 정리했다.
학습 종료 후 **🏆 전체 최고 노선**을 누르면 실행 전체에서 보상이 가장 높았던 경로를 지도에 고정한다. 화면 없이 돌린 실험도 `python3 train.py --replay runs/dqn-42`로 열 수 있다.

### 최종 비교 지도에 도로 경로 표시

학습이 끝난 비교의 우승 노선들만 OpenStreetMap 기반 FOSSGIS OSRM 차량 라우터로 조회한다.

```sh
python3 route_geometry.py runs/road-comparison
python3 train.py --replay runs/road-comparison
```

라우팅은 인터넷이 필요하다. 같은 좌표로 실행하면 저장된 결과를 재사용한다.
최종 지도에는 차량 라우팅 거리와 학습에 사용한 도로거리 합계가 각각 표시된다.
일반 차량 라우터이므로 버스의 실제 회전 및 정차 가능성은 별도로 확인한다.
별도 스크립트로 받은 경로는 최종 비교 지도에 적용된다.

### 학습 중 연속 도로거리로 제한하기

차량용 OSRM을 서울을 포함하는 OSM 데이터로 로컬에서 실행한 뒤 주소를 지정한다.
[OSRM 설치·차량 그래프 준비 안내](https://github.com/Project-OSRM/osrm-backend#quick-start)

```sh
python3 train.py --osrm http://127.0.0.1:5000 --budget-m 10000 --out runs/osrm
```

다음 정류장을 추가한 전체 경유 순서를 차량 라우터에 전달한다. 해당 경로가 없거나 10km를 초과하면 후보에서 제외한다. 가능한 후보가 없으면 현재 정류장에서 종료한다. 보상의 거리 비용은 연속 경로 거리의 변화량으로 계산한다. 결과 화면에는 마지막 판정과 같은 거리가 반환된 도로 형상을 사용한다. 라우터 없이 행렬 진단을 실행하려면 `--data network_role4_diagnostic.json`을 지정한다. 이전 282곳 실험은 `--data network.json`으로 재현한다.

라우터는 별도 프로세스다. 공개 서버는 학습에 사용하지 않는다. 경로 캐시는 실행 중 네 방법이 공유하며, 새 경유 순서마다 후보 정류장까지의 경로를 계산하므로 행렬 사용보다 느리다. 짧은 확인에는 `--algo random --episodes 1 --rounds-per-episode 2`를 추가한다.
