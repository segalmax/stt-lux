#!/usr/bin/env python3
"""
World Cup 2026 — bookies-grounded EV picks + heatmaps for 2026-06-21 (Jerusalem).
Source: bet365 odds via kickoff.co.uk (fetched 2026-06-21 ~07:00 IDT).
All matches GROUP STAGE: direction 1 pt, exact 3 pts -> EV = P(class) + 2*P(exact).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Group stage scoring
DIR_PTS, EXACT_PTS = 1, 3

MATCHES = [
    {
        "home": "Spain", "away": "Saudi Arabia", "ko": "19:00 IDT",
        "x2": {"home": 1.09, "draw": 10.00, "away": 26.00},
        # correct-score decimal odds, keyed by (home_goals, away_goals)
        "cs": {
            (2, 0): 4.50, (3, 0): 6.00, (4, 0): 7.50, (1, 0): 8.00,
            (5, 0): 13.00, (2, 1): 15.00, (3, 1): 15.00, (4, 1): 19.00,
            (1, 1): 21.00, (0, 0): 17.00, (6, 0): 23.00, (5, 1): 29.00,
            (0, 1): 41.00,  # Saudi 1-0  (away win)
            (7, 0): 41.00, (6, 1): 41.00,
        },
    },
    {
        "home": "Belgium", "away": "Iran", "ko": "22:00 IDT",
        "x2": {"home": 1.42, "draw": 4.50, "away": 8.00},
        "cs": {
            (2, 0): 6.50, (1, 0): 6.50, (2, 1): 9.00, (1, 1): 9.00,
            (3, 0): 9.50, (3, 1): 13.00, (0, 0): 11.00,
            (0, 1): 17.00,  # Iran 1-0
            (4, 0): 19.00, (2, 2): 23.00,
            (1, 2): 23.00,  # Iran 2-1
            (4, 1): 23.00, (3, 2): 34.00, (5, 0): 41.00,
            (0, 2): 41.00,  # Iran 2-0
        },
    },
]

STAGE_LABEL = "group"


def devig_1x2(x2):
    imp = {k: 1.0 / v for k, v in x2.items()}
    s = sum(imp.values())
    return {k: v / s for k, v in imp.items()}


def classify(hg, ag):
    if hg > ag:
        return "home"
    if hg < ag:
        return "away"
    return "draw"


def devig_cs(cs, p1x2):
    """De-vig correct-score within each outcome class so each class sums to its 1X2 prob."""
    classes = {"home": [], "draw": [], "away": []}
    for (hg, ag), odds in cs.items():
        classes[classify(hg, ag)].append(((hg, ag), 1.0 / odds))
    p = {}
    for cls, items in classes.items():
        tot = sum(imp for _, imp in items)
        if tot == 0:
            continue
        for score, imp in items:
            p[score] = imp / tot * p1x2[cls]
    return p


def analyze(m):
    p1x2 = devig_1x2(m["x2"])
    pcs = devig_cs(m["cs"], p1x2)
    # favorite = higher win prob
    if p1x2["home"] >= p1x2["away"]:
        fav, dog, fav_side = m["home"], m["away"], "home"
    else:
        fav, dog, fav_side = m["away"], m["home"], "away"
    p_fav_win = p1x2[fav_side]

    rows = []  # (fav_g, dog_g, p_exact, ev)
    for (hg, ag), pex in pcs.items():
        cls = classify(hg, ag)
        pclass = p1x2[cls]
        ev = DIR_PTS * pclass + (EXACT_PTS - DIR_PTS) * pex
        # orient to favorite axis
        if fav_side == "home":
            fg, dg = hg, ag
        else:
            fg, dg = ag, hg
        label = f"{m['home']} {hg}-{ag}" if cls != "draw" else f"Draw {hg}-{ag}"
        rows.append({"fg": fg, "dg": dg, "hg": hg, "ag": ag, "cls": cls,
                     "pex": pex, "ev": ev, "label": label})
    rows.sort(key=lambda r: r["ev"], reverse=True)
    return {"m": m, "p1x2": p1x2, "pcs": pcs, "fav": fav, "dog": dog,
            "fav_side": fav_side, "p_fav_win": p_fav_win, "rows": rows}


def build_grid(a):
    """EV and P(exact) grids: y=fav goals 0..5, x=dog goals 0..5."""
    p1x2 = a["p1x2"]
    fav_side = a["fav_side"]
    ev = np.zeros((6, 6))
    pex = np.zeros((6, 6))
    for fg in range(6):
        for dg in range(6):
            # map fav/dog goals back to home/away
            if fav_side == "home":
                hg, ag = fg, dg
            else:
                hg, ag = dg, fg
            cls = classify(hg, ag)
            p_exact = a["pcs"].get((hg, ag), 0.0)
            pclass = p1x2[cls]
            ev[fg, dg] = DIR_PTS * pclass + (EXACT_PTS - DIR_PTS) * p_exact
            pex[fg, dg] = p_exact
    return ev, pex


def heatmap(a, path):
    ev, pex = build_grid(a)
    fav, dog = a["fav"], a["dog"]
    best = max(((fg, dg) for fg in range(6) for dg in range(6)),
              key=lambda t: ev[t[0], t[1]])
    bfg, bdg = best

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(ev, origin="lower", cmap="YlOrRd", aspect="equal")
    for fg in range(6):
        for dg in range(6):
            ax.text(dg, fg + 0.16, f"{ev[fg, dg]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="black")
            ax.text(dg, fg - 0.22, f"{pex[fg, dg]*100:.1f}%", ha="center", va="center",
                    fontsize=8, color="dimgray")
    # blue box on best
    ax.add_patch(Rectangle((bdg - 0.5, bfg - 0.5), 1, 1, fill=False,
                           edgecolor="blue", linewidth=3))
    # dotted draw diagonal
    for i in range(6):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="black", linewidth=1.2, linestyle=":"))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel(f"{dog} goals (underdog)")
    ax.set_ylabel(f"{fav} goals (favorite)")
    ax.set_title(f"{fav} v {dog} — best: {fav} {bfg}-{bdg} (EV {ev[bfg, bdg]:.2f})")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("EV")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return (bfg, bdg, ev[bfg, bdg])


def main():
    summary = []
    for m in MATCHES:
        a = analyze(m)
        slug = f"{m['home']}_{m['away']}".lower().replace(" ", "-")
        path = os.path.join(OUTDIR, f"heatmap_{slug}.png")
        bfg, bdg, bev = heatmap(a, path)
        fav, dog = a["fav"], a["dog"]

        print("=" * 64)
        print(f"{m['home']} v {m['away']}  (KO {m['ko']})  — {STAGE_LABEL} stage")
        print(f"  De-vig 1X2: {m['home']} {a['p1x2']['home']*100:.1f}% | "
              f"Draw {a['p1x2']['draw']*100:.1f}% | "
              f"{m['away']} {a['p1x2']['away']*100:.1f}%")
        print(f"  Favorite: {fav} ({a['p_fav_win']*100:.1f}% win)")
        print(f"  Top 6 scorelines by EV:")
        print(f"    {'scoreline':<18}{'P(exact)':>10}{'EV':>8}")
        for r in a["rows"][:6]:
            print(f"    {r['label']:<18}{r['pex']*100:>9.1f}%{r['ev']:>8.3f}")
        best = a["rows"][0]
        best_draw = next(r for r in a["rows"] if r["cls"] == "draw")
        print(f"  >> BEST PICK: {best['label']}  (EV {best['ev']:.3f}, "
              f"P(exact) {best['pex']*100:.1f}%)")
        print(f"  >> Best DRAW: {best_draw['label']}  (EV {best_draw['ev']:.3f}, "
              f"P(exact) {best_draw['pex']*100:.1f}%)")
        print(f"  heatmap -> {os.path.basename(path)}  (best cell {fav} {bfg}-{bdg} EV {bev:.2f})")
        summary.append({"match": f"{m['home']} v {m['away']}", "ko": m["ko"],
                        "pick": best["label"], "ev": best["ev"],
                        "draw": best_draw["label"], "draw_ev": best_draw["ev"]})

    print("\n" + "=" * 64)
    print("SUMMARY — best pick per match")
    print(f"  {'match':<26}{'KO':<12}{'best pick':<18}{'EV':>6}")
    for s in summary:
        print(f"  {s['match']:<26}{s['ko']:<12}{s['pick']:<18}{s['ev']:>6.3f}")
    print("\nCaveats: bet365 correct-score list is partial (~15 lines, no 'any other "
          "score' bucket); de-vigged within each outcome class from listed scores only. "
          "Saudi Arabia win class had a single listed scoreline (1-0).")


if __name__ == "__main__":
    main()
