# 연세대 연계 마을버스 노선 설계

첨부 자료의 후보 40개와 방향별 도로거리 행렬을 바탕으로 정류장 방문 순서를 학습한다.
`network.json`은 자료 02·13·14번을 정류장 ID로 연결한 모델 입력이다. 후보와 거리 행렬이 갱신되면 `python3 prepare.py "/path/to/data-folder"` 로 다시 만든다. 좌표와 거리의 출처는 팀원 제공 자료이며, 배경 지도는 OpenStreetMap이다.

## 실행

Python 3.9 이상에서 Q-learning, SARSA, 기준 전략을 바로 실행할 수 있다.
DQN은 PyTorch가 필요하다. GPU 머신에는 CUDA를 지원하는 PyTorch를 설치한다.

```sh
python3 train.py --algo q-learning --steps 3000
python3 -m http.server 8765 --bind 127.0.0.1
```

브라우저에서 http://127.0.0.1:8765/ 를 연다. 별도 터미널에서 같은 폴더의 학습을 실행하면 에피소드 기록이 갱신되고, 버스 아이콘이 기록된 정류장 순서를 재생한다. 지도 타일과 Leaflet에는 인터넷 연결이 필요하다. 선과 아이콘의 이동은 정류장 좌표를 직선으로 연결하며, 보상에 쓰는 거리는 도로거리 행렬 값이다.

```sh
python3 train.py --algo sarsa --steps 3000 --out runs/sarsa
python3 -m pip install -r requirements.txt
python3 train.py --algo dqn --steps 5000 --device cpu --out runs/dqn
python3 train.py --algo random --episodes 100 --out runs/random
python3 train.py --algo greedy --episodes 1 --out runs/greedy
python3 train.py --algo dp --max-stops 8 --episodes 1 --out runs/dp
```

다른 실행 기록은 주소 뒤에 `?run=dqn`처럼 출력 폴더명을 붙여 확인한다. 실험별로 별도의 출력 폴더를 지정한다.
대규모 비교 시 세 RL 방식에 각각 동일한 `--steps 200000`, 설정, 시드 목록을 적용한다. CUDA에서는 `--device cuda`를 사용한다.
DP 최적해와 비교할 때는 모든 방식에 같은 `--max-stops`를 지정한다. 후보 제한은 출발점·종점과 후보표 앞부분을 사용한다.

## 환경과 보상

- 상태는 현재 정류장, 방문 집합, 남은 거리다. 다음 정류장을 선택하면 방향별 도로거리를 차감한다.
- 도로거리 총합이 상한을 넘거나, 그 정류장을 들른 뒤 종점에 도달할 수 없는 행동을 선택지에서 제외한다. 재정차도 제외한다.
- 수요 점수는 5–8월 오전 7–9시 월간 하차량을 월 일수로 나누고, 월별 일평균의 평균을 후보 내 최댓값으로 나눈다. 9–10월 값은 학습된 노선의 월별 경유지 하차량 평가에 사용한다.
- 점수 = 경유 정류장별 정규화 하차량 + 최초 거점 방문당 2점 − 도로거리 1km당 0.2점. 출발과 종점의 하차 점수는 고정이므로 제외한다.
- 시작점은 연대앞.세브란스병원(112900188), 종점은 서대문역3번출구(100900044)를 임시로 지정했다. 고정 출발 거점인 연대앞은 방문 처리되어 보너스가 추가되지 않는다. 이대·신촌·아현은 각 최초 정차에 2점이다. 연대앞까지 선택형으로 비교하려면 `--start`로 일반 정류장을 지정한다.
- 거점은 대표 정류장 ID로 정의했다. `prepare.py`에서 수정 가능하다. 기본 거리 상한은 출발→종점 도로거리의 1.6배다. `--ratio`, `--bonus`, `--cost`, `--start`, `--goal`로 조정한다.
- 하차량은 기존 이용 실적이다. 거리 자료는 실제 마을버스의 회전 및 정차 가능성까지 검증한 값으로 해석하지 않는다.

## 비교 방식과 출력

Q-learning과 SARSA는 동일한 상태 표현으로 각각 다음 상태의 최대 Q와 다음에 실제 선택한 행동의 Q를 사용한다. DQN은 128–128 MLP, 경험 재생, 타깃 네트워크와 유효 행동 마스크를 사용한다. 세 방법 모두 `--exploration epsilon-greedy` 또는 `--exploration greedy`를 선택한다. 기준 탐욕법은 이용량 점수와 신규 거점 보너스의 합을 이번 이동거리로 나눈다. DP는 작은 후보 집합에서 벨만 최적값을 메모이제이션한다.

`history.jsonl`에는 학습 중 에피소드, `trace.json`에는 최근 기록·최고 탐험 점수·탐험을 끈 현재 정책을 저장한다. 모델은 `model.pt` 또는 `q_table.json`에 저장된다.
`--steps`에서는 마지막 노선을 종점까지 완성하되 요청한 전이 횟수에서 모델 업데이트를 멈춘다.
