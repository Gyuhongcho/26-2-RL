"""Train a bus route policy and publish episode traces for the local viewer."""
import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import deque
from functools import lru_cache
import json
import math
from pathlib import Path
import random
from threading import Thread
import time
import webbrowser

ROOT = Path(__file__).resolve().parent


class Route:
    def __init__(self, data, budget, bonus, cost):
        self.stops = data["stops"]
        self.distance = data["distances_m"]
        self.budget = budget
        self.bonus, self.cost = bonus, cost
        self.monthly = {month: [n / stats["days"] for n in stats["alight"]]
                        for month, stats in data["months"].items()}
        train = [v for month, v in self.monthly.items() if data["months"][month]["split"] == "train"]
        self.demand = [sum(x) / len(train) for x in zip(*train)]
        self.scale = max(1, max(self.demand))
        self.train_months = {m for m, v in data["months"].items() if v["split"] == "train"}

    def hubs(self, path):
        return {self.stops[i]["hub"] for i in path} - {""}

    def actions(self, state):
        current, visited, meters = state
        return [i for i, distance in enumerate(self.distance[current])
                if not (visited >> i) & 1
                and meters + distance <= self.budget + 1e-7]

    def step(self, state, action):
        current, visited, meters = state
        distance = self.distance[current][action]
        gain = self.gain(visited, action)
        reward = gain - self.cost * distance / 1000
        return (action, visited | (1 << action), round(meters + distance, 3)), reward

    def gain(self, visited, action):
        hub = self.stops[action]["hub"]
        prior_hubs = self.hubs(i for i in range(len(self.stops)) if (visited >> i) & 1)
        new_hub = hub and hub not in prior_hubs
        demand = self.demand[action] / self.scale
        return demand + self.bonus * bool(new_hub)

    def features(self, state):
        current, visited, meters = state
        n = len(self.stops)
        return ([float(i == current) for i in range(n)]
                + [float((visited >> i) & 1) for i in range(n)]
                + [1 - meters / self.budget])

    def summary(self, path, rewards):
        return dict(path=path, rewards=rewards, score=sum(rewards),
                    distance_m=sum(self.distance[a][b] for a, b in zip(path, path[1:])),
                    hubs=sorted(self.hubs(path)), complete=True,
                    validation={m: sum(values[i] for i in path[1:])
                                for m, values in self.monthly.items() if m not in self.train_months})


class TableAgent:
    def __init__(self, method, alpha, gamma):
        self.q = {}
        self.method, self.alpha, self.gamma = method, alpha, gamma

    def value(self, state, action):
        # Round only the Q-table key to avoid float addition artifacts.
        return self.q.get((state[0], state[1], round(state[2], 3), action), 0.0)

    def best(self, state, actions):
        return max(actions, key=lambda action: self.value(state, action))

    def update(self, state, action, reward, following, actions, next_action):
        future = 0
        if actions:
            selected = next_action if self.method == "sarsa" else self.best(following, actions)
            future = self.value(following, selected)
        old = self.value(state, action)
        self.q[state[0], state[1], round(state[2], 3), action] = old + self.alpha * (reward + self.gamma * future - old)

    def save(self, path):
        publish(path, [[*key, value] for key, value in self.q.items()])


class DQN:
    def __init__(self, route, args, rng):
        import torch
        self.torch = torch
        self.route, self.rng = route, rng
        self.device = args.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(args.seed)
        if self.device == "cpu":
            torch.set_num_threads(1)
        n = len(route.stops)
        self.net = torch.nn.Sequential(torch.nn.Linear(2*n+1, 128), torch.nn.ReLU(),
                                       torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                       torch.nn.Linear(128, n)).to(self.device)
        import copy
        self.target = copy.deepcopy(self.net).eval()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.memory = deque(maxlen=50000)
        self.gamma = args.gamma
        self.updates = 0

    def tensor(self, values, **kwargs):
        return self.torch.tensor(values, device=self.device, **kwargs)

    def best(self, state, actions):
        with self.torch.no_grad():
            scores = self.net(self.tensor(self.route.features(state))).tolist()
        return max(actions, key=scores.__getitem__)

    def update(self, state, action, reward, following, actions, next_action):
        self.memory.append((self.route.features(state), action, reward,
                            self.route.features(following), actions))
        if len(self.memory) < 1000:
            return
        torch = self.torch
        s, a, r, nxt, valid = zip(*self.rng.sample(self.memory, 64))
        allowed = self.tensor([[i in actions for i in range(len(self.route.stops))]
                               for actions in valid], dtype=torch.bool)
        with torch.no_grad():
            next_q = self.target(self.tensor(nxt)).masked_fill(~allowed, -torch.inf).max(1).values
            expected = self.tensor(r) + self.gamma * next_q.masked_fill(~allowed.any(1), 0)
        actual = self.net(self.tensor(s)).gather(1, self.tensor(a).unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(actual, expected)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 1000 == 0:
            self.target.load_state_dict(self.net.state_dict())

    def save(self, path):
        self.torch.save(self.net.state_dict(), path)


def exact_policy(route, gamma):
    @lru_cache(maxsize=None)
    def solve(state):
        actions = route.actions(state)
        if not actions:
            return 0.0, None
        def score(action):
            following, reward = route.step(state, action)
            return reward + gamma * solve(following)[0]
        action = max(actions, key=score)
        return score(action), action
    return lambda state, actions: solve(state)[1]


def episode(route, start, choose, learn=None):
    state = (start, 1 << start, 0)
    path, rewards = [start], []
    actions = route.actions(state)
    action = choose(state, actions) if actions else None
    while action is not None:
        following, reward = route.step(state, action)
        actions = route.actions(following)
        next_action = choose(following, actions) if actions else None
        if learn is not None:
            learn(state, action, reward, following, actions, next_action)
        path.append(action)
        rewards.append(reward)
        state, action = following, next_action
    return route.summary(path, rewards)


def publish(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


@contextmanager
def viewer(enabled, output):
    if not enabled:
        yield
        return

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            files = {"/": (ROOT / "index.html", "text/html"),
                     "/trace.json": (output / "trace.json", "application/json")}
            item = files.get(self.path.partition("?")[0])
            if item is None:
                self.send_error(404)
                return
            path, content_type = item
            try:
                payload = path.read_bytes()
            except FileNotFoundError:
                self.send_response(204)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), ViewerHandler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"지도 시뮬레이터: {url}", flush=True)
        try:
            webbrowser.open(url)
            yield
            print("학습 완료. 지도는 계속 재생됩니다. 종료하려면 Ctrl+C.", flush=True)
            while True:
                time.sleep(1)
        finally:
            server.shutdown()
            worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["q-learning", "sarsa", "dqn", "random", "greedy", "dp"], default="q-learning")
    parser.add_argument("--exploration", choices=["epsilon-greedy", "greedy"], default="epsilon-greedy")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--rounds-per-episode", type=int, default=20)
    parser.add_argument("--steps", type=int, help="Exact training transition budget")
    parser.add_argument("--max-stops", type=int, help="Limit candidates for small exact comparisons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="DQN only: auto, cpu, cuda or mps")
    parser.add_argument("--data", type=Path, default=ROOT / "network.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "local")
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--budget-m", type=float, default=6000)
    parser.add_argument("--bonus", type=float, default=2)
    parser.add_argument("--cost", type=float, default=0.2, help="Reward deduction per km")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--no-viewer", action="store_true", help="Train without opening the local map")
    args = parser.parse_args()
    if (min(args.episodes, args.rounds_per_episode) < 1 or not math.isfinite(args.budget_m) or args.budget_m <= 0
            or not all(math.isfinite(x) and x >= 0 for x in (args.bonus, args.cost))
            or not 0 < args.alpha <= 1 or not 0 <= args.gamma <= 1):
        parser.error("Invalid episodes, distance budget, reward weights, alpha or gamma")
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if args.steps is not None and (args.steps < 1 or args.algo not in {"q-learning", "sarsa", "dqn"}):
        parser.error("--steps requires a positive budget and a learning algorithm")
    if args.max_stops is not None:
        if args.max_stops < 5:
            parser.error("--max-stops must be at least 5")
        indices = list(range(min(args.max_stops, len(data["stops"]))))
        data["stops"] = [data["stops"][i] for i in indices]
        data["distances_m"] = [[data["distances_m"][i][j] for j in indices] for i in indices]
        for month in data["months"].values():
            month["alight"] = [month["alight"][i] for i in indices]
    if args.algo == "dp" and len(data["stops"]) > 12:
        parser.error("Use --max-stops 5 to 12 for the exact DP experiment")
    route = Route(data, args.budget_m, args.bonus, args.cost)
    eligible = [i for i, stop in enumerate(route.stops)
                if not stop["hub"] and route.actions((i, 1 << i, 0))]
    if len(eligible) < 5:
        parser.error("Need five non-hub stops with an available first move; increase --max-stops or --budget-m")
    starts = random.Random(args.start_seed).sample(eligible, 5)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    start_rng = random.Random(args.start_seed)
    agent = None
    if args.algo == "dqn":
        agent = DQN(route, args, rng)
    elif args.algo in {"q-learning", "sarsa"}:
        agent = TableAgent(args.algo, args.alpha, args.gamma)

    dp = exact_policy(route, args.gamma) if args.algo == "dp" else None
    steps = 0

    def learn(*transition):
        nonlocal steps
        if args.steps is None or steps < args.steps:
            agent.update(*transition)
            steps += 1

    def greedy(state, actions):
        return min(actions, key=lambda i: (
            -route.gain(state[1], i) / max(1, route.distance[state[0]][i]), route.stops[i]["id"]))

    def choose(state, actions):
        if dp:
            return dp(state, actions)
        fraction = steps / args.steps if args.steps else (i - 1) / total_rounds
        epsilon = max(0.05, 1 - 0.95 * fraction / 0.7) if args.exploration == "epsilon-greedy" else 0
        if args.algo == "random" or (agent and rng.random() < epsilon):
            return rng.choice(actions)
        return agent.best(state, actions) if agent else greedy(state, actions)

    recent = deque(maxlen=40)
    winners = []
    episode_best = None
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    trace = dict(stops=route.stops, daily_alight=route.demand, budget_m=route.budget, config=config, distances_m=route.distance)
    trace["starts"] = [route.stops[i] for i in starts]
    publish(args.out / "starts.json", dict(seed=args.start_seed, stops=trace["starts"]))
    total_rounds = args.episodes * args.rounds_per_episode
    stride = max(1, total_rounds // 200)
    publish(args.out / "trace.json", dict(trace, recent=[], progress=0, steps=0, finished=False))
    publish(args.out / "episode_winners.json", winners)
    with viewer(not args.no_viewer, args.out.resolve()):
        with (args.out / "history.jsonl").open("w", encoding="utf-8") as log:
            i = 0
            while steps < args.steps if args.steps else i < total_rounds:
                i += 1
                episode_number = (i - 1) // args.rounds_per_episode + 1
                start = start_rng.choice(starts)
                run = dict(round=i, episode=episode_number, start_id=route.stops[start]["id"],
                           **episode(route, start, choose, learn if agent else None))
                run["steps"] = steps
                log.write(json.dumps(run, ensure_ascii=False) + "\n")
                if episode_best is None or run["score"] > episode_best["score"]:
                    episode_best = run
                recent.append(run)
                finished = steps >= args.steps if args.steps else i == total_rounds
                episode_end = i % args.rounds_per_episode == 0 or finished
                if episode_end:
                    winners.append(dict(episode_best, rounds=(i - 1) % args.rounds_per_episode + 1))
                    publish(args.out / "episode_winners.json", winners)
                    episode_best = None
                if not args.no_viewer and (i == 1 or i % stride == 0 or episode_end) or finished:
                    publish(args.out / "trace.json", dict(trace, recent=list(recent), best=winners[-1:],
                                                         winners=winners[-40:], progress=i,
                                                         steps=steps, finished=finished))
                    if not args.no_viewer and not finished:
                        time.sleep(0.08)
            if agent:
                agent.save(args.out / ("model.pt" if args.algo == "dqn" else "q_table.json"))
        print(f"{args.algo}: {i} rounds / {steps} training steps | {len(winners)} episode winners | {args.out}")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n실행을 종료했습니다.")
