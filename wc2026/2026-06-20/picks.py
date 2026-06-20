#!/usr/bin/env python3
"""WC2026 bookies-grounded EV picks + heatmaps for 2026-06-20.

Data: bet365 odds via kickoff.co.uk. Group stage scoring:
  direction pts = 1, exact pts = 3  ->  EV = P(class) + 2*P(exact).
De-vig 1X2 by normalizing implied (1/odds) to sum 1.
De-vig correct-score by normalizing implied within each outcome class
so each class sums to its 1X2 probability. The fetched CS list omits a
small "any other score" tail, so de-vig from what is listed (caveat).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))

# Group stage scoring
DIR_PTS, EXACT_PTS = 1, 3

MATCHES = [
    {
        "home": "Netherlands", "away": "Sweden", "ko": "20:00 IDT",
        "h2x2": {"home": 1.70, "draw": 4.10, "away": 4.50},
        # scoreline -> odds. h-a relative to home(Netherlands)-away(Sweden)
        "cs": {
            (1, 1): 8.00, (2, 1): 9.00, (1, 0): 8.50, (2, 0): 9.00,
            (3, 1): 13.00, (1, 2): 15.00, (0, 1): 15.00, (3, 0): 15.00,
            (2, 2): 15.00, (0, 0): 13.00, (3, 2): 23.00, (0, 2): 26.00,
            (4, 1): 26.00, (4, 0): 29.00, (1, 3): 41.00,
        },
    },
    {
        "home": "Germany", "away": "Ivory Coast", "ko": "23:00 IDT",
        "h2x2": {"home": 1.53, "draw": 3.50, "away": 4.50},
        "cs": {
            (2, 0): 9.00, (2, 1): 8.50, (1, 0): 9.00, (1, 1): 9.00,
            (3, 0): 12.00, (3, 1): 12.00, (2, 2): 17.00, (1, 2): 19.00,
            (0, 0): 15.00, (0, 1): 19.00, (4, 0): 21.00, (4, 1): 21.00,
            (3, 2): 23.00, (0, 2): 34.00, (4, 2): 41.00,
        },
    },
]


def devig_1x2(o):
    imp = {k: 1.0 / v for k, v in o.items()}
    s = sum(imp.values())
    return {k: v / s for k, v in imp.items()}


def cls_of(h, a):
    return "home" if h > a else ("away" if h < a else "draw")


def devig_cs(cs, p1x2):
    """Normalize implied CS probs within each class to that class's 1X2 prob."""
    imp = {sc: 1.0 / od for sc, od in cs.items()}
    cls_sum = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for sc, p in imp.items():
        cls_sum[cls_of(*sc)] += p
    out = {}
    for sc, p in imp.items():
        c = cls_of(*sc)
        out[sc] = p / cls_sum[c] * p1x2[c] if cls_sum[c] > 0 else 0.0
    return out


def analyze(m):
    p1x2 = devig_1x2(m["h2x2"])
    pcs = devig_cs(m["cs"], p1x2)
    # favorite by win prob
    fav_home = p1x2["home"] >= p1x2["away"]
    fav_team = m["home"] if fav_home else m["away"]
    dog_team = m["away"] if fav_home else m["home"]
    fav_winprob = p1x2["home"] if fav_home else p1x2["away"]

    rows = []  # (scoreline_str, fav_goals, dog_goals, p_exact, ev)
    for (h, a), p in pcs.items():
        c = cls_of(h, a)
        pclass = p1x2[c]
        ev = DIR_PTS * pclass + (EXACT_PTS - DIR_PTS) * p
        # orient to fav/dog
        if fav_home:
            fg, dg = h, a
        else:
            fg, dg = a, h
        sl = f"{m['home']} {h}-{a}" if c == "home" else (
            f"{m['away']} {a}-{h}" if c == "away" else f"Draw {h}-{a}")
        rows.append({"sl": sl, "h": h, "a": a, "fg": fg, "dg": dg,
                     "cls": c, "p": p, "ev": ev})
    rows.sort(key=lambda r: r["ev"], reverse=True)
    return {"m": m, "p1x2": p1x2, "pcs": pcs, "fav": fav_team, "dog": dog_team,
            "fav_home": fav_home, "fav_winprob": fav_winprob, "rows": rows}


def heatmap(res, idx):
    m = res["m"]
    fav, dog = res["fav"], res["dog"]
    fav_home = res["fav_home"]
    N = 6  # 0..5
    EV = np.zeros((N, N))
    PX = np.zeros((N, N))
    for fg in range(N):
        for dg in range(N):
            h, a = (fg, dg) if fav_home else (dg, fg)
            p = res["pcs"].get((h, a), 0.0)
            c = cls_of(h, a)
            pclass = res["p1x2"][c]
            # If scoreline not priced, P(exact)=0 -> EV=direction only for its class
            EV[fg, dg] = DIR_PTS * pclass + (EXACT_PTS - DIR_PTS) * p
            PX[fg, dg] = p

    best = res["rows"][0]
    bfg, bdg = best["fg"], best["dg"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(EV, origin="lower", cmap="YlOrRd", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("EV (points)")

    for fg in range(N):
        for dg in range(N):
            ax.text(dg, fg + 0.18, f"{EV[fg, dg]:.2f}", ha="center",
                    va="center", fontsize=10, fontweight="bold", color="black")
            ax.text(dg, fg - 0.22, f"{PX[fg, dg] * 100:.1f}%", ha="center",
                    va="center", fontsize=7, color="#333333")

    # blue box around best EV cell
    ax.add_patch(Rectangle((bdg - 0.5, bfg - 0.5), 1, 1, fill=False,
                           edgecolor="blue", linewidth=3))
    # dotted draw diagonal
    for d in range(N):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                               edgecolor="black", linewidth=1.2,
                               linestyle=(0, (2, 2))))

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xlabel(f"{dog} goals (underdog)")
    ax.set_ylabel(f"{fav} goals (favorite)")
    sl_fav = best["fg"]
    ax.set_title(f"{fav} v {dog} — best: {fav} {best['fg']}-{best['dg']} "
                 f"(EV {best['ev']:.2f})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT, f"heatmap_{idx}_{fav.lower().replace(' ', '')}_"
                        f"{dog.lower().replace(' ', '')}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    summary = []
    for i, m in enumerate(MATCHES, 1):
        res = analyze(m)
        path = heatmap(res, i)
        print("=" * 64)
        print(f"MATCH {i}: {m['home']} v {m['away']}  (KO {m['ko']}) — group stage")
        p = res["p1x2"]
        print(f"  De-vig 1X2: {m['home']} {p['home']*100:.1f}% | "
              f"Draw {p['draw']*100:.1f}% | {m['away']} {p['away']*100:.1f}%")
        print(f"  Favorite: {res['fav']} (win {res['fav_winprob']*100:.1f}%)")
        print(f"  Top 6 by EV:")
        print(f"    {'scoreline':<20}{'P(exact)':>10}{'EV':>8}")
        for r in res["rows"][:6]:
            print(f"    {r['sl']:<20}{r['p']*100:>9.1f}%{r['ev']:>8.2f}")
        best = res["rows"][0]
        best_draw = next(r for r in res["rows"] if r["cls"] == "draw")
        print(f"  BEST PICK: {best['sl']}  (P {best['p']*100:.1f}%, EV {best['ev']:.2f})")
        print(f"  BEST DRAW: {best_draw['sl']}  (P {best_draw['p']*100:.1f}%, "
              f"EV {best_draw['ev']:.2f})")
        print(f"  heatmap -> {os.path.basename(path)}")
        summary.append((m, res, best, best_draw, path))

    print("\n" + "=" * 64)
    print("SUMMARY — best pick per match")
    print(f"{'Match':<30}{'Best pick':<22}{'EV':>6}{'KO':>12}")
    for m, res, best, bd, _ in summary:
        label = f"{m['home']} v {m['away']}"
        print(f"{label:<30}{best['sl']:<22}{best['ev']:>6.2f}{m['ko']:>12}")
    return summary


if __name__ == "__main__":
    main()
