"""AlphaNLHoldem inference worker. Runs under the Python 3.7 / TF1.15 / Ray 0.8.3 env from poker-ai PR #2.
Protocol: prints {"ready": true} once loaded; then for each stdin line (JSON obs with card_info,
action_info, extra_info, legal_moves as nested lists) prints {"action": <0..4>}.
Usage: python eval/alphanl_worker.py --repo /path/to/poker-ai [--weights path/to/c_1048.pkl]
"""
import argparse, json, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, os.path.join(a.repo, "src"))
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # keep library prints off the protocol channel
    import numpy as np
    from alphanl.inference import AlphaNLHoldemAgent
    agent = AlphaNLHoldemAgent(weights_path=a.weights)
    def out(d):
        real_stdout.write(json.dumps(d) + "\n"); real_stdout.flush()
    out({"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        o = json.loads(line)
        obs = {"card_info": np.asarray(o["card_info"], np.uint8),
               "action_info": np.asarray(o["action_info"], np.uint8),
               "extra_info": np.asarray(o["extra_info"], np.uint8),
               "legal_moves": np.asarray(o["legal_moves"], np.uint8)}
        out({"action": int(agent.make_action(obs))})

if __name__ == "__main__":
    main()
