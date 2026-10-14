"""Slumbot API client (heads-up NLHE, blinds 50/100, 200bb = 20000 chips).
Based on the official sample client at slumbot.com. Action string grammar:
  k=check c=call f=fold bN=bet/raise so that this street's total bet is N; '/' ends a street.

Examples:
  python eval/slumbot_client.py --hands 100 --agent random --stack-bb 50 --log runs/random.jsonl
  python eval/slumbot_client.py --hands 50000 --agent alphanl --stack-bb 50 --workers 4 \
      --log runs/alphanl.jsonl --alphanl-python /path/to/py37/python --alphanl-repo /path/to/poker-ai
  python eval/slumbot_client.py --summarize runs/alphanl.jsonl
Re-running the same command resumes: completed hands in --log are counted and skipped.
"""
import argparse, json, math, os, random, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import requests

HOST = "https://slumbot.com"
SB, BB, STACK = 50, 100, 20000
MAX_WORKERS = 4  # hard cap to stay friendly with Slumbot rate limits
HALF_POT_MAX, POT_MAX = 0.75, 1.6  # bet/pot ratio buckets for mapping Slumbot bets to abstract actions
RL_CHIP = BB // 2  # AlphaNLHoldem/rlcard: BB = 2 chips, 50bb = 100 chips -> 1 rlcard chip = 50 Slumbot chips


# ----------------------------------------------------------------------------- game state
class State:
    """Parse a Slumbot action string. pos 0 = BB, pos 1 = SB (SB acts first preflop, BB first postflop).
    Also records an event list [(street, pos, kind, street_total_after)] for history encoding."""
    def __init__(self, action):
        self.street, self.pos = 0, 1
        self.contrib = [BB, SB]
        self.street_last = BB
        self.prev = [0, 0]
        self.over = False
        self.events = []
        i = 0
        while i < len(action):
            ch = action[i]
            if ch == "/":
                self.prev = [self.prev[p] + self.contrib[p] for p in (0, 1)]
                self.contrib = [0, 0]; self.street += 1; self.pos = 0; self.street_last = BB; i += 1; continue
            p = self.pos
            before = [self.prev[0] + self.contrib[0], self.prev[1] + self.contrib[1]]
            if ch == "k":
                kind = "k"; i += 1
            elif ch == "c":
                self.contrib[p] = self.contrib[p ^ 1]; kind = "c"; i += 1
            elif ch == "f":
                self.over = True; kind = "f"; i += 1
            elif ch == "b":
                j = i + 1
                while j < len(action) and action[j].isdigit(): j += 1
                amt = int(action[i + 1:j])
                self.street_last = max(amt - max(self.contrib), BB)
                self.contrib[p] = amt; kind = "b"; i = j
            else:
                raise ValueError(f"bad action {action!r}")
            self.events.append((self.street, p, kind, before, [self.prev[0] + self.contrib[0], self.prev[1] + self.contrib[1]]))
            self.pos ^= 1

    def committed(self, p):
        return self.prev[p] + self.contrib[p]

    def legal(self):
        me, opp = self.pos, self.pos ^ 1
        to_call = self.contrib[opp] - self.contrib[me]
        stack_left = STACK - self.committed(me)
        acts = ["f", "c"] if to_call > 0 else ["k"]
        max_total = self.contrib[me] + stack_left
        min_total = min(self.contrib[opp] + self.street_last, max_total)
        if stack_left > to_call and STACK - self.committed(opp) > 0:
            acts.append(("b", min_total, max_total))
        return acts, to_call, stack_left


# ----------------------------------------------------------------------------- 50bb view
def capped_legal(st, legal, cap):
    """Slumbot-format legal actions clipped to an effective stack of `cap` chips for BOTH players.
    Returns (legal, view_truncated). If opponent is already at/over cap, only fold/call(/check)."""
    if cap is None:
        return legal, False
    me, opp = st.pos, st.pos ^ 1
    trunc = st.committed(opp) > cap or st.committed(me) >= cap
    room = cap - st.committed(me)
    out = []
    for a in legal:
        if isinstance(a, tuple):
            if st.committed(opp) >= cap: continue          # opp all-in in our view: no raise
            _, lo, hi = a
            hi2 = min(hi, st.contrib[me] + max(room, 0))
            if hi2 >= lo: out.append(("b", lo, hi2))
        else:
            out.append(a)
    return out, trunc


def abstract_obs(st, hole, board, cap):
    """Build the AlphaNLHoldem (rlcard no-limit-holdem 1.1.0) observation features as plain lists.
    Actions: 0 fold, 1 check/call, 2 raise half pot, 3 raise pot, 4 all-in.
    Player id = Slumbot pos (0 = BB, 1 = SB). Chips are in rlcard units with stacks capped to `cap`."""
    capc = cap if cap is not None else STACK
    V = lambda x: min(x, capc)
    history = [[], [], [], []]
    def legal_at(c_me, c_opp, s_me, s_opp):
        # c_* = total committed (view), s_* = this-street committed (view), rlcard units = chips / RL_CHIP
        pot = (c_me + c_opp) / RL_CHIP
        remain = (capc - c_me) / RL_CHIP
        diff = (s_opp - s_me) / RL_CHIP
        acts = [0, 1]
        if diff > 0 and diff >= remain:
            return acts
        if remain > 0: acts.append(4)
        if int(pot / 2) <= remain and int(pot / 2) + s_me / RL_CHIP > max(s_me, s_opp) / RL_CHIP: acts.insert(2, 2)
        if pot <= remain: acts.insert(len(acts) - (1 if 4 in acts else 0), 3)
        return sorted(set(acts))
    street_start = [0, 0]; cur_street = 0
    for (street, p, kind, before, after) in st.events:
        if street != cur_street:
            street_start = [V(before[0]), V(before[1])]; cur_street = street
        cb, ca = [V(x) for x in before], [V(x) for x in after]
        s_me, s_opp = cb[p] - street_start[p], cb[p ^ 1] - street_start[p ^ 1]
        la = legal_at(cb[p], cb[p ^ 1], s_me, s_opp)
        if kind == "f": a = 0
        elif kind in "kc": a = 1
        else:
            added = ca[p] - cb[p]; pot = cb[0] + cb[1]
            if ca[p] >= capc: a = 4
            else:
                r = added / max(pot, 1)
                a = 2 if r < HALF_POT_MAX else 3 if r < POT_MAX else 4
            if a not in la:
                cand = [x for x in (2, 3, 4) if x in la]
                a = min(cand, key=lambda x: abs(x - a)) if cand else 1
        history[street].append([p, a, la])
    me = st.pos
    c = [V(st.committed(0)), V(st.committed(1))]
    street_start = [V(st.prev[0]), V(st.prev[1])]
    legal = legal_at(c[me], c[me ^ 1], c[me] - street_start[me], c[me ^ 1] - street_start[me ^ 1])
    # card_info [4 suits][13 ranks][6]: hand, flop, turn, river, public, all
    suit = dict(zip("cdhs", range(4))); rank = dict(zip("23456789TJQKA", range(13)))
    card = [[[0] * 6 for _ in range(13)] for _ in range(4)]
    def put(cs, ch):
        for x in cs: card[suit[x[1].lower()]][rank[x[0].upper()]][ch] = 1
    put(hole, 0); put(board[:3], 1); put(board[3:4], 2); put(board[4:5], 3); put(board, 4); put(board + hole, 5)
    act = [[[0] * 25 for _ in range(5)] for _ in range(4)]
    for ri, h in enumerate(history):
        for hi, (pid, aid, la) in enumerate(h[:6]):
            act[pid][aid][ri * 6 + hi] = 1; act[2][aid][ri * 6 + hi] = 1
            for l in la: act[3][l][ri * 6 + hi] = 1
    for a in range(4):
        for b in range(5): act[a][b][24] = me
    stakes = [int((capc - c[0]) / RL_CHIP), int((capc - c[1]) / RL_CHIP)]
    lm = [1 if i in legal else 0 for i in range(5)]
    return {"card_info": card, "action_info": act, "extra_info": stakes, "legal_moves": lm}, legal


def abstract_to_slumbot(aid, st, legal, cap):
    """Map an abstract action id to a Slumbot action string, clamped to the (capped) legal range."""
    me, opp = st.pos, st.pos ^ 1
    can_call = "c" in legal
    if aid == 0:
        return "f" if can_call else "k"
    raise_opt = next((a for a in legal if isinstance(a, tuple)), None)
    if aid == 1 or raise_opt is None:
        return "c" if can_call else "k"
    capc = cap if cap is not None else STACK
    pot = min(st.committed(0), capc) + min(st.committed(1), capc)
    if aid == 2: total = st.contrib[me] + pot // 2
    elif aid == 3: total = st.contrib[me] + pot
    else: total = raise_opt[2]
    _, lo, hi = raise_opt
    return f"b{max(lo, min(hi, total))}"


# ----------------------------------------------------------------------------- agents
class Agent:
    """Swappable agent interface. act(obs) returns 'f','k','c' or 'b<N>' (N = street total).
    obs keys: action, hole_cards, board, client_pos, legal (capped), to_call, stack_left, state, abstract, abstract_legal."""
    def act(self, obs):
        raise NotImplementedError
    def close(self):
        pass


class RandomAgent(Agent):
    def __init__(self, seed=None): self.rng = random.Random(seed)
    def act(self, obs):
        choice = self.rng.choice(obs["legal"])
        if isinstance(choice, tuple):
            _, lo, hi = choice
            return f"b{self.rng.randint(lo, hi)}"
        return choice


class CallAgent(Agent):
    def act(self, obs):
        return "c" if "c" in obs["legal"] else "k"


class AbstractRandomAgent(Agent):
    """Random over the 5 AlphaNLHoldem abstract actions; exercises the same mapping path as the model."""
    def __init__(self, seed=None): self.rng = random.Random(seed)
    def act(self, obs):
        return self.rng.choice(obs["abstract_legal"])


class AlphaNLAgent(Agent):
    """AlphaNLHoldem in a Python 3.7 / TF1.15 subprocess (eval/alphanl_worker.py), JSON lines over stdin/stdout."""
    def __init__(self, python, repo, weights=None):
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alphanl_worker.py")
        cmd = [python, worker, "--repo", repo] + (["--weights", weights] if weights else [])
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.p.stdout.readline()
        if not ready.startswith("{") or not json.loads(ready).get("ready"):
            raise RuntimeError(f"alphanl worker failed to start: {ready!r}")
        self.lock = threading.Lock()

    def act(self, obs):
        with self.lock:
            self.p.stdin.write(json.dumps(obs["abstract"]) + "\n"); self.p.stdin.flush()
            line = self.p.stdout.readline()
        if not line: raise RuntimeError("alphanl worker died")
        aid = int(json.loads(line)["action"])
        if aid not in obs["abstract_legal"]:
            aid = 1  # model picked an action illegal in our view: check/call
        return aid

    def close(self):
        try: self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception: self.p.kill()


def make_agent(args, idx):
    seed = None if args.seed is None else args.seed * 1000 + idx
    if args.agent == "random": return RandomAgent(seed)
    if args.agent == "call": return CallAgent()
    if args.agent == "abstract-random": return AbstractRandomAgent(seed)
    if args.agent == "alphanl":
        if not args.alphanl_repo: sys.exit("--alphanl-repo is required for --agent alphanl")
        return AlphaNLAgent(args.alphanl_python, args.alphanl_repo, args.alphanl_weights)
    raise ValueError(args.agent)


# ----------------------------------------------------------------------------- HTTP
class RateLimiter:
    def __init__(self, min_interval):
        self.min_interval, self.lock, self.next = min_interval, threading.Lock(), 0.0
    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next); self.next = t + self.min_interval
        if t > now: time.sleep(t - now)


LIMITER = RateLimiter(0.2)


def post(session, path, data, retries=8):
    for k in range(retries):
        LIMITER.wait()
        try:
            r = session.post(f"{HOST}/api/{path}", json=data, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            j = r.json()
            if "error_msg" in j:
                raise RuntimeError(j["error_msg"])
            return j
        except (requests.RequestException, ValueError) as e:
            if k == retries - 1: raise
            time.sleep(min(60, 2 ** k) + random.random())


def play_hand(session, agent, cap, token=None):
    r = post(session, "new_hand", {"token": token} if token else {})
    trunc = False
    while True:
        token = r.get("token", token)
        if "winnings" in r:
            st = State(r["action"])
            deep = max(st.committed(0), st.committed(1)) > cap if cap else False
            return r, token, trunc, deep
        st = State(r["action"])
        legal, to_call, stack_left = st.legal()
        clegal, t = capped_legal(st, legal, cap)
        trunc = trunc or t
        ab, ab_legal = abstract_obs(st, r["hole_cards"], r["board"], cap)
        obs = dict(action=r["action"], hole_cards=r["hole_cards"], board=r["board"], client_pos=r["client_pos"],
                   legal=clegal, to_call=to_call, stack_left=min(stack_left, cap - st.committed(st.pos)) if cap else stack_left,
                   state=st, abstract=ab, abstract_legal=ab_legal)
        a = agent.act(obs)
        if isinstance(a, int):
            a = abstract_to_slumbot(a, st, clegal, cap)
        r = post(session, "act", {"token": token, "incr": a})


# ----------------------------------------------------------------------------- run / summary
def count_done(path):
    if not os.path.exists(path): return 0
    n = 0
    with open(path) as f:
        for line in f:
            try: json.loads(line); n += 1
            except ValueError: pass  # partial last line from a crash
    return n


def stats(xs):
    n = len(xs)
    if n == 0: return {"hands": 0}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1))
    k = 1000 / BB
    return {"hands": n, "mbb_per_hand": round(m * k, 1), "ci95": round(1.96 * sd / math.sqrt(n) * k, 1)}


def summarize(path):
    rows = [json.loads(l) for l in open(path) if l.strip().startswith("{")]
    w = lambda f: [r["winnings"] for r in rows if f(r)]
    out = {"baseline_not_truncated": stats(w(lambda r: not r.get("truncated"))),
           "note": "BASELINE = non-truncated hands. Truncated hands are systematically disadvantaged for the 50bb agent "
                   "(it is capped while Slumbot plays 200bb); report them separately, not as the headline number.",
           "all": stats(w(lambda r: True)),
           "pos": {name: stats(w(lambda r, p=p: r["client_pos"] == p)) for p, name in ((0, "BB"), (1, "SB"))},
           "truncated": stats(w(lambda r: r.get("truncated"))),
           "past_50bb_rate": round(sum(1 for r in rows if r.get("past_cap")) / max(len(rows), 1), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=100, help="TOTAL target hands in --log (resume-aware)")
    ap.add_argument("--agent", default="random", choices=["random", "call", "abstract-random", "alphanl"])
    ap.add_argument("--stack-bb", type=float, default=50, help="effective stack cap for both players (0 = off)")
    ap.add_argument("--log", default="slumbot_hands.jsonl")
    ap.add_argument("--half-pot-max", type=float, default=0.75, help="Slumbot bet/pot ratio below this -> half-pot action")
    ap.add_argument("--pot-max", type=float, default=1.6, help="ratio below this -> pot action, else all-in")
    ap.add_argument("--workers", type=int, default=2, help=f"concurrent sessions (max {MAX_WORKERS})")
    ap.add_argument("--min-interval", type=float, default=0.2, help="global min seconds between API requests")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alphanl-python", default=os.environ.get("ALPHANL_PYTHON", "python3.7"))
    ap.add_argument("--alphanl-repo", default=os.environ.get("ALPHANL_REPO"))
    ap.add_argument("--alphanl-weights", default=None)
    ap.add_argument("--summarize", metavar="JSONL")
    a = ap.parse_args()
    if a.summarize:
        print(json.dumps(summarize(a.summarize), indent=2)); return
    LIMITER.min_interval = a.min_interval
    global HALF_POT_MAX, POT_MAX
    HALF_POT_MAX, POT_MAX = a.half_pot_max, a.pot_max
    workers = max(1, min(a.workers, MAX_WORKERS))
    cap = int(a.stack_bb * BB) if a.stack_bb else None
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    done = count_done(a.log)
    todo = a.hands - done
    print(f"[resume] {done} hands already in {a.log}; playing {max(todo, 0)} more with {workers} worker(s)", flush=True)
    if todo <= 0:
        print(json.dumps(summarize(a.log))); return
    lock = threading.Lock(); counter = [done]; remaining = [todo]; t0 = time.time()
    # drop a half-written trailing line left by a crash so appends stay valid jsonl
    if os.path.exists(a.log):
        with open(a.log, "rb+") as f:
            data = f.read()
            if data and not data.endswith(b"\n"):
                f.truncate(data.rfind(b"\n") + 1)
    fout = open(a.log, "a")

    def run(idx):
        agent = make_agent(a, idx); session = requests.Session(); token = None
        try:
            while True:
                with lock:
                    if remaining[0] <= 0: return
                    remaining[0] -= 1
                try:
                    r, token, trunc, deep = play_hand(session, agent, cap, token)
                except Exception as e:
                    print(f"[w{idx}] hand failed ({e}); new session", file=sys.stderr, flush=True)
                    with lock: remaining[0] += 1
                    token = None; time.sleep(5); continue
                with lock:
                    rec = {"hand": counter[0], "worker": idx, "client_pos": r["client_pos"], "action": r["action"],
                           "hole": r["hole_cards"], "bot_hole": r.get("bot_hole_cards"), "board": r["board"],
                           "winnings": r["winnings"], "truncated": trunc, "past_cap": deep, "cap": cap, "agent": a.agent,
                           "ts": round(time.time(), 1)}
                    fout.write(json.dumps(rec) + "\n"); fout.flush(); os.fsync(fout.fileno())
                    counter[0] += 1
                    n = counter[0] - done
                    if n % 50 == 0:
                        print(f"[progress] {counter[0]}/{a.hands} ({n / (time.time() - t0):.2f} hands/s)", flush=True)
        finally:
            agent.close()

    try:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(run, range(workers)))
    finally:
        fout.close()
    print(json.dumps(summarize(a.log)))


if __name__ == "__main__":
    main()
