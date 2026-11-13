"""Train a bus route policy and publish episode traces for the local viewer."""
import argparse
from collections import deque
from functools import lru_cache
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


class Route:
    def __init__(self, data, start, goal, ratio, bonus, cost):
        self.stops = data["stops"]
        ids = [stop["id"] for stop in self.stops]
        self.start, self.goal = ids.index(start), ids.index(goal)
        self.distance = data["distances_m"]
        direct = self.distance[self.start][self.goal]
        if self.start == self.goal or direct <= 0:
            raise ValueError("Choose distinct endpoints with a positive distance")
        self.budget = direct * ratio
        self.bonus, self.cost = bonus, cost
        self.monthly = {month: [n / stats["days"] for n in stats["alight"]]
                        for month, stats in data["months"].items()}
        train = [v for month, v in self.monthly.items() if data["months"][month]["split"] == "train"]
        self.demand = [sum(x) / len(train) for x in zip(*train)]
        self.scale = max(1, max(self.demand))
        self.fixed_hubs = self.hubs([self.start, self.goal])
        self.train_months = {m for m, v in data["months"].items() if v["split"] == "train"}

    def hubs(self, path):
        return {self.stops[i]["hub"] for i in path} - {""}

    def actions(self, state):
        current, visited, meters = state
        return [i for i, distance in enumerate(self.distance[current])
                if not (visited >> i) & 1
                and meters + distance + self.distance[i][self.goal] <= self.budget + 1e-7]

    def step(self, state, action):
        current, visited, meters = state
        distance = self.distance[current][action]
        gain = self.gain(visited, action)
        reward = gain - self.cost * distance / 1000
        return (action, visited | (1 << action), round(meters + distance, 3)), reward

    def gain(self, visited, action):
        hub = self.stops[action]["hub"]
        prior_hubs = self.hubs(i for i in range(len(self.stops)) if (visited >> i) & 1)
        new_hub = hub and hub not in self.fixed_hubs and hub not in prior_hubs
        demand = 0 if action == self.goal else self.demand[action] / self.scale
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
                    hubs=sorted(self.hubs(path)), complete=path[-1] == self.goal,
                    validation={m: sum(values[i] for i in path[1:-1])
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
        if state[0] == route.goal:
            return 0.0, None
        def score(action):
            following, reward = route.step(state, action)
            return reward + gamma * solve(following)[0]
        action = max(route.actions(state), key=score)
        return score(action), action
    return lambda state, actions: solve(state)[1]


def episode(route, choose, learn=None):
    state = (route.start, 1 << route.start, 0)
    path, rewards = [route.start], []
    action = choose(state, route.actions(state))
    while state[0] != route.goal:
        following, reward = route.step(state, action)
        actions = [] if action == route.goal else route.actions(following)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["q-learning", "sarsa", "dqn", "random", "greedy", "dp"], default="q-learning")
    parser.add_argument("--exploration", choices=["epsilon-greedy", "greedy"], default="epsilon-greedy")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, help="Exact training transition budget")
    parser.add_argument("--max-stops", type=int, help="Limit candidates for small exact comparisons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="DQN only: auto, cpu, cuda or mps")
    parser.add_argument("--data", type=Path, default=ROOT / "network.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "local")
    parser.add_argument("--start", default="112900188")
    parser.add_argument("--goal", default="100900044")
    parser.add_argument("--ratio", type=float, default=1.6)
    parser.add_argument("--bonus", type=float, default=2)
    parser.add_argument("--cost", type=float, default=0.2, help="Reward deduction per km")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1)
    args = parser.parse_args()
    if (args.episodes < 1 or not math.isfinite(args.ratio) or args.ratio < 1
            or not all(math.isfinite(x) and x >= 0 for x in (args.bonus, args.cost))
            or not 0 < args.alpha <= 1 or not 0 <= args.gamma <= 1):
        parser.error("Invalid episodes, ratio, reward weights, alpha or gamma")
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if args.steps is not None and (args.steps < 1 or args.algo not in {"q-learning", "sarsa", "dqn"}):
        parser.error("--steps requires a positive budget and a learning algorithm")
    if args.max_stops is not None:
        if args.max_stops < 2:
            parser.error("--max-stops must be at least 2")
        endpoints = [i for i, stop in enumerate(data["stops"]) if stop["id"] in {args.start, args.goal}]
        others = [i for i in range(len(data["stops"])) if i not in endpoints]
        indices = sorted(endpoints + others[:args.max_stops - len(endpoints)])
        data["stops"] = [data["stops"][i] for i in indices]
        data["distances_m"] = [[data["distances_m"][i][j] for j in indices] for i in indices]
        for month in data["months"].values():
            month["alight"] = [month["alight"][i] for i in indices]
    if args.algo == "dp" and len(data["stops"]) > 12:
        parser.error("Use --max-stops 5 to 12 for the exact DP experiment")
    route = Route(data, args.start, args.goal, args.ratio, args.bonus, args.cost)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
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
        nonterminal = [i for i in actions if i != route.goal]
        if not nonterminal:
            return route.goal
        return min(nonterminal, key=lambda i: (
            -route.gain(state[1], i) / max(1, route.distance[state[0]][i]), route.stops[i]["id"]))

    def choose(state, actions):
        if dp:
            return dp(state, actions)
        fraction = steps / args.steps if args.steps else (i - 1) / args.episodes
        epsilon = max(0.05, 1 - 0.95 * fraction / 0.7) if args.exploration == "epsilon-greedy" else 0
        if args.algo == "random" or (agent and rng.random() < epsilon):
            return rng.choice(actions)
        return agent.best(state, actions) if agent else greedy(state, actions)

    recent = deque(maxlen=40)
    best = None
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    trace = dict(stops=route.stops, daily_alight=route.demand, budget_m=route.budget, config=config, distances_m=route.distance)
    stride = max(1, args.episodes // 200)
    with (args.out / "history.jsonl").open("w", encoding="utf-8") as log:
        i = 0
        while steps < args.steps if args.steps else i < args.episodes:
            i += 1
            run = dict(episode=i, **episode(route, choose, learn if agent else None))
            run["steps"] = steps
            log.write(json.dumps(run, ensure_ascii=False) + "\n")
            if best is None or run["score"] > best["score"]:
                best = run
            finished = steps >= args.steps if args.steps else i == args.episodes
            if i == 1 or i % stride == 0 or finished:
                recent.append(run)
                policy = episode(route, agent.best) if agent else run
                publish(args.out / "trace.json", dict(trace, recent=list(recent), best=best,
                                                     policy=policy, progress=i, steps=steps, finished=finished))
        if agent:
            agent.save(args.out / ("model.pt" if args.algo == "dqn" else "q_table.json"))
    print(f"{args.algo}: {i} episodes / {steps} training steps | best sampled {best['score']:.3f} | final policy {policy['score']:.3f} | {args.out}")


if __name__ == "__main__":
    main()
