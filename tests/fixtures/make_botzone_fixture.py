"""Compact public Botzone TexasHoldem2p match logs into tests/fixtures/botzone_real.jsonl.gz.

Input: JSON files with {"log": <_rawLogJSON>, "initdata": ...} scraped from
https://www.botzone.org.cn/match/<id> (public, no login).
One line per hand: m=match id, h=hand no, d=dealer_id, c=[cards p0, cards p1]
(Botzone ints, rank*4+suit, suit order hearts/diamonds/spades/clubs),
b=board known from the log (5 cards whenever the hand reached showdown),
a=[[player_id, response, street at decision], ...], t=judge action_type per
action (None when the log's last display omits it), r=[win_chips p0, p1].
"""
import collections, glob, gzip, json, os, sys

def main(src_glob, dst):
    out = []
    for f in sorted(glob.glob(src_glob)):
        mid = os.path.basename(f)[:-5]
        L = json.load(open(f))["log"]
        T = []
        for i in range(0, len(L) - 2, 2):
            j, b, j2 = L[i], L[i + 1], L[i + 2]
            if "output" not in j or j["output"]["command"] != "request":
                continue
            pid = list(j["output"]["content"])[0]
            T.append((int(pid), j["output"]["content"][pid], b.get(pid, {}).get("response"),
                      j["output"]["display"], j2["output"]["display"]))
        by = collections.OrderedDict()
        for t in T:
            by.setdefault(t[1]["hand"], []).append(t)
        for hno, ts in by.items():
            end = ts[-1][4]
            seen = [t[1]["public_cards"] for t in ts] + [end.get("last_public_cards") or []]
            if end.get("temp_result") and end.get("last_public_cards") is None:
                seen.append(end["public_cards"])
            out.append({"m": mid, "h": hno, "d": ts[0][1]["dealer_id"], "c": ts[0][3]["player_cards"],
                        "b": max(seen, key=len), "a": [[t[0], t[2], t[3]["round"]] for t in ts],
                        "t": [(t[4].get("last_action") or {}).get("action_type") for t in ts],
                        "r": [x["win_chips"] for x in end["temp_result"]]})
    with gzip.open(dst, "wt") as g:
        for r in out:
            g.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(len(out), "hands ->", dst, os.path.getsize(dst), "bytes")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
