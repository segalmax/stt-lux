#!/usr/bin/env python3
"""WC2026 bookies-grounded EV picks + heatmaps for 2026-06-20 (IDT).

Bet365 odds via kickoff.co.uk. De-vig 1X2 -> P(class); de-vig correct-score
WITHIN each outcome class so each class sums to its 1X2 prob. Group-stage
scoring: direction 1 pt / exact 3 pts -> EV = P(class) + 2*P(exact).
Correct-score list is partial (no "any other score" bucket); de-vigged from
listed scorelines only -- caveat noted.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Group-stage scoring
DIR_PTS, EXACT_PTS = 1, 3  # EV = DIR*P(class) + (EXACT-DIR)*P(exact)

MATCHES = [
    {
        "home": "Netherlands", "away": "Sweden",
        "ko": "20:00 IDT", "stage": "group",
        "x12": {"home": 1.70, "draw": 4.00, "away": 4.75},
        # (home_goals, away_goals, bet365 decimal odds)
        "cs": [
            (1, 1, 8.00), (2, 1, 9.00), (1, 0, 9.50), (2, 0, 9.50),
            (3, 1, 13.00), (1, 2, 15.00), (2, 2, 15.00), (3, 0, 15.00),
            (0, 1, 17.00), (0, 0, 15.00), (3, 2, 23.00), (0, 2, 26.00),
            (4, 1, 26.00), (4, 0, 26.00), (1, 3, 34.00),
        ],
    },
    {
        "home": "Germany", "away": "Ivory Coast",
        "ko": "23:00 IDT", "stage": "group",
        "x12": {"home": 1.50, "draw": 4.50, "away": 5.75},
        "cs": [
            (2, 1, 8.50), (2, 0, 9.00), (1, 1, 9.00), (1, 0, 9.50),
            (3, 1, 12.00), (3, 0, 12.00), (2, 2, 15.00), (1, 2, 19.00),
            (0, 1, 21.00), (0, 0, 17.00), (4, 1, 21.00), (4, 0, 21.00),
            (3, 2, 21.00), (0, 2, 34.00), (4, 2, 34.00),
        ],
    },
]


def devig_1x2(x12):
    imp = {k: 1.0 / v for k, v in x12.items()}
    s = sum(imp.values())
    return {k: v / s for k, v in imp.items()}


def classify(hg, ag):
    if hg > ag:
        return "home"
    if hg < ag:
        return "away"
    return "draw"


def devig_cs(cs, p_class):
    """De-vig correct-score within each outcome class to its 1X2 prob."""
    # raw implied per class
    by_cls = {"home": [], "draw": [], "away": []}
    for hg, ag, odds in cs:
        by_cls[classify(hg, ag)].append((hg, ag, 1.0 / odds))
    p_exact = {}
    for cls, items in by_cls.items():
        tot = sum(p for _, _, p in items)
        if tot == 0:
            continue
        for hg, ag, p in items:
            p_exact[(hg, ag)] = (p / tot) * p_class[cls]
    return p_exact


def analyze(m):
    p_class = devig_1x2(m["x12"])
    p_exact = devig_cs(m["cs"], p_class)

    # favorite by win prob
    fav_is_home = p_class["home"] >= p_class["away"]
    fav = m["home"] if fav_is_home else m["away"]
    dog = m["away"] if fav_is_home else m["home"]
    p_favwin = p_class["home"] if fav_is_home else p_class["away"]
    p_dogwin = p_class["away"] if fav_is_home else p_class["home"]

    rows = []  # (fav_goals, dog_goals, p_exact, ev, label)
    for hg, ag, _ in m["cs"]:
        pe = p_exact[(hg, ag)]
        cls = classify(hg, ag)
        if cls == "draw":
            pcls = p_class["draw"]
        elif (cls == "home") == fav_is_home:
            pcls = p_favwin
        else:
            pcls = p_dogwin
        ev = DIR_PTS * pcls + (EXACT_PTS - DIR_PTS) * pe
        fg, dg = (hg, ag) if fav_is_home else (ag, hg)
        rows.append((fg, dg, pe, ev, f"{fav} {fg}-{dg}" if cls != "draw"
                     else f"Draw {fg}-{dg}"))
    rows.sort(key=lambda r: -r[3])
    return {
        "fav": fav, "dog": dog, "fav_is_home": fav_is_home,
        "p_class": p_class, "p_exact": p_exact, "p_favwin": p_favwin,
        "p_dogwin": p_dogwin, "rows": rows, "m": m,
    }


def ev_grid(a):
    """EV[y=fav goals, x=dog goals] for 0..5, using class+exact probs."""
    fav_is_home = a["fav_is_home"]
    grid = np.zeros((6, 6))
    pe_grid = np.zeros((6, 6))
    for fg in range(6):
        for dg in range(6):
            hg, ag = (fg, dg) if fav_is_home else (dg, fg)
            pe = a["p_exact"].get((hg, ag), 0.0)
            cls = classify(hg, ag)
            if cls == "draw":
                pcls = a["p_class"]["draw"]
            elif (cls == "home") == fav_is_home:
                pcls = a["p_favwin"]
            else:
                pcls = a["p_dogwin"]
            grid[fg, dg] = DIR_PTS * pcls + (EXACT_PTS - DIR_PTS) * pe
            pe_grid[fg, dg] = pe
    return grid, pe_grid


def heatmap(a, path):
    grid, pe = ev_grid(a)
    best = np.unravel_index(np.argmax(grid), grid.shape)
    by, bx = best
    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(grid, cmap="YlOrRd", origin="lower", aspect="equal")
    for fg in range(6):
        for dg in range(6):
            ax.text(dg, fg + 0.16, f"{grid[fg, dg]:.2f}", ha="center",
                    va="center", fontsize=10, fontweight="bold")
            ax.text(dg, fg - 0.22, f"{pe[fg, dg]*100:.1f}%", ha="center",
                    va="center", fontsize=7, color="#333")
    # blue box around best
    ax.add_patch(Rectangle((bx - 0.5, by - 0.5), 1, 1, fill=False,
                 edgecolor="blue", lw=3))
    # dotted draw diagonal
    for d in range(6):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                     edgecolor="black", lw=1.4, ls=":"))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel(f"{a['dog']} goals (underdog)")
    ax.set_ylabel(f"{a['fav']} goals (favorite)")
    ax.set_title(f"{a['fav']} v {a['dog']} — best: {a['fav']} "
                 f"{by}-{bx} (EV {grid[by, bx]:.2f})", fontweight="bold")
    fig.colorbar(im, ax=ax, label="EV (points)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return (by, bx, grid[by, bx])


def main():
    summary = []
    for m in MATCHES:
        a = analyze(m)
        fname = f"{m['home'].lower().replace(' ', '-')}-vs-" \
                f"{m['away'].lower().replace(' ', '-')}.png"
        path = os.path.join(OUTDIR, fname)
        by, bx, bev = heatmap(a, path)

        pc = a["p_class"]
        print(f"\n=== {m['home']} v {m['away']}  (KO {m['ko']}, "
              f"{m['stage']} stage) ===")
        print(f"De-vig 1X2: {m['home']} {pc['home']*100:.1f}% | "
              f"Draw {pc['draw']*100:.1f}% | {m['away']} {pc['away']*100:.1f}%")
        print(f"Favorite: {a['fav']} (win {a['p_favwin']*100:.1f}%)")
        print(f"{'Scoreline':<20}{'P(exact)':>10}{'EV':>8}")
        for fg, dg, pe, ev, lab in a["rows"][:6]:
            print(f"{lab:<20}{pe*100:>9.1f}%{ev:>8.2f}")
        best = a["rows"][0]
        draws = [r for r in a["rows"] if "Draw" in r[4]]
        bd = draws[0]
        print(f"BEST PICK : {best[4]}  (EV {best[3]:.2f}, "
              f"P {best[2]*100:.1f}%)")
        print(f"BEST DRAW : {bd[4]}  (EV {bd[3]:.2f}, P {bd[2]*100:.1f}%)")
        print(f"heatmap -> {fname}")
        summary.append((a, best, bd, fname))

    print("\n\n===== SUMMARY (all matches) =====")
    print(f"{'Match':<28}{'Best pick':<20}{'EV':>6}{'  KO':>10}")
    for a, best, bd, _ in summary:
        m = a["m"]
        print(f"{m['home']+' v '+m['away']:<28}{best[4]:<20}"
              f"{best[3]:>6.2f}  {m['ko']:>8}")
    print("\nCaveat: bet365 correct-score lists are partial (~15 lines, no "
          "'any other score' bucket); de-vigged within class from listed "
          "scorelines only.")


if __name__ == "__main__":
    main()
