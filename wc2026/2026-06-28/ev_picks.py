#!/usr/bin/env python3
"""World Cup 2026 bookies-grounded EV picks + heatmaps.

Data: bet365 odds via kickoff.co.uk (fetched manually, hardcoded below).
Date: 2026-06-28 (Asia/Jerusalem). 48h forward window.
All 3 priced matches are Round of 32 -> direction 2 pts, exact 5 pts.
EV = 2*P(class) + (5-2)*P(exact) = 2*P(class) + 3*P(exact)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Round of 32 scoring
DIR_PTS = 2
EXACT_PTS = 5

# Each match: home/away, 1X2 decimal odds, correct-score list as (home_goals, away_goals, odds)
MATCHES = [
    {
        "home": "South Africa", "away": "Canada",
        "ko": "22:00 IDT Sun Jun 28", "stage": "R32",
        "o_home": 5.25, "o_draw": 3.60, "o_away": 1.70,
        "cs": [
            (0,1,6.00),(0,2,7.50),(1,1,7.00),(1,2,9.50),(0,0,8.00),
            (1,0,12.00),(0,3,13.00),(1,3,17.00),(2,1,19.00),(2,2,21.00),
            (2,0,29.00),(0,4,29.00),(1,4,34.00),(2,3,34.00),(3,1,51.00),
        ],
    },
    {
        "home": "Brazil", "away": "Japan",
        "ko": "20:00 IDT Mon Jun 29", "stage": "R32",
        "o_home": 1.67, "o_draw": 2.80, "o_away": 5.00,
        "cs": [
            (1,0,6.50),(1,1,7.50),(2,0,8.00),(2,1,9.00),(0,0,9.50),
            (0,1,13.00),(3,0,13.00),(3,1,15.00),(1,2,19.00),(2,2,19.00),
            (0,2,29.00),(4,0,26.00),(3,2,29.00),(4,1,29.00),(1,3,41.00),
        ],
    },
    {
        "home": "Netherlands", "away": "Morocco",
        "ko": "04:00 IDT Tue Jun 30", "stage": "R32",
        "o_home": 2.10, "o_draw": 3.20, "o_away": 2.80,
        "cs": [
            (1,1,6.00),(1,0,7.00),(0,1,10.00),(0,0,8.00),(2,1,8.50),
            (2,0,10.00),(1,2,15.00),(0,2,19.00),(2,2,17.00),(3,1,19.00),
            (3,0,21.00),(1,3,34.00),(3,2,34.00),(0,3,41.00),(2,3,41.00),
        ],
    },
]


def devig_1x2(oh, od, oa):
    imp = np.array([1/oh, 1/od, 1/oa])
    return imp / imp.sum()  # P(home), P(draw), P(away)


def devig_cs(cs, p_home, p_draw, p_away):
    """De-vig correct-score within each outcome class so each class sums to its 1X2 prob."""
    classes = {"home": [], "draw": [], "away": []}
    for hg, ag, odd in cs:
        if hg > ag: classes["home"].append((hg, ag, 1/odd))
        elif hg == ag: classes["draw"].append((hg, ag, 1/odd))
        else: classes["away"].append((hg, ag, 1/odd))
    target = {"home": p_home, "draw": p_draw, "away": p_away}
    p_exact = {}  # (hg,ag) -> prob
    for cls, items in classes.items():
        s = sum(i[2] for i in items)
        if s == 0: continue
        for hg, ag, imp in items:
            p_exact[(hg, ag)] = imp / s * target[cls]
    return p_exact


def analyse(m):
    p_home, p_draw, p_away = devig_1x2(m["o_home"], m["o_draw"], m["o_away"])
    p_exact = devig_cs(m["cs"], p_home, p_draw, p_away)

    # favorite by win prob
    if p_home >= p_away:
        fav, dog, fav_is_home = m["home"], m["away"], True
        p_fav, p_dog = p_home, p_away
    else:
        fav, dog, fav_is_home = m["away"], m["home"], False
        p_fav, p_dog = p_away, p_home

    # Build EV/Pexact grids: y = favorite goals (0..5), x = underdog goals (0..5)
    N = 6
    EV = np.zeros((N, N))
    PE = np.zeros((N, N))
    for fg in range(N):
        for dg in range(N):
            # map fav/dog goals back to home/away
            if fav_is_home:
                hg, ag = fg, dg
            else:
                hg, ag = dg, fg
            # outcome class
            if hg > ag:
                p_cls = p_home
            elif hg == ag:
                p_cls = p_draw
            else:
                p_cls = p_away
            pe = p_exact.get((hg, ag), 0.0)
            PE[fg, dg] = pe
            EV[fg, dg] = DIR_PTS * p_cls + (EXACT_PTS - DIR_PTS) * pe

    return {
        "m": m, "fav": fav, "dog": dog, "fav_is_home": fav_is_home,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_fav": p_fav, "p_dog": p_dog,
        "p_exact": p_exact, "EV": EV, "PE": PE,
    }


def ranked_table(r):
    """Top scorelines by EV across listed scorelines (cells)."""
    rows = []
    N = 6
    for fg in range(N):
        for dg in range(N):
            ev = r["EV"][fg, dg]
            pe = r["PE"][fg, dg]
            rows.append((fg, dg, pe, ev))
    rows.sort(key=lambda x: -x[3])
    return rows


def heatmap(r):
    m = r["m"]; EV = r["EV"]; PE = r["PE"]
    fav, dog = r["fav"], r["dog"]
    N = 6
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(EV, origin="lower", cmap="YlOrRd", aspect="equal")

    # best cell
    bi = np.unravel_index(np.argmax(EV), EV.shape)
    best_fg, best_dg = bi

    for fg in range(N):
        for dg in range(N):
            ev = EV[fg, dg]; pe = PE[fg, dg]*100
            ax.text(dg, fg+0.13, f"{ev:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="black")
            ax.text(dg, fg-0.22, f"{pe:.1f}%", ha="center", va="center",
                    fontsize=8, color="dimgray")

    # blue box around best EV
    ax.add_patch(Rectangle((best_dg-0.5, best_fg-0.5), 1, 1, fill=False,
                           edgecolor="blue", lw=3))
    # dotted outline on draw diagonal
    for d in range(N):
        ax.add_patch(Rectangle((d-0.5, d-0.5), 1, 1, fill=False,
                               edgecolor="black", lw=1.5, linestyle=":"))

    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xlabel(f"{dog} goals (underdog)", fontsize=12)
    ax.set_ylabel(f"{fav} goals (favorite)", fontsize=12)
    ax.set_title(f"{fav} v {dog} — best: {fav} {best_fg}-{best_dg} (EV {EV[bi]:.2f})\n"
                 f"{m['stage']}  •  KO {m['ko']}", fontsize=13)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("EV (points)")
    fig.tight_layout()
    slug = f"{fav}_v_{dog}".lower().replace(" ", "-")
    path = os.path.join(OUTDIR, f"heatmap_{slug}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path, (best_fg, best_dg, EV[bi])


def main():
    summary = []
    for m in MATCHES:
        r = analyse(m)
        path, best = heatmap(r)
        rows = ranked_table(r)
        fav, dog = r["fav"], r["dog"]
        print("\n" + "="*64)
        print(f"{fav} v {dog}  [{m['stage']}]  KO {m['ko']}")
        print(f"  De-vig 1X2:  {m['home']} {r['p_home']*100:.1f}%  "
              f"Draw {r['p_draw']*100:.1f}%  {m['away']} {r['p_away']*100:.1f}%")
        print(f"  Favorite: {fav} ({r['p_fav']*100:.1f}% win)")
        print(f"  Top 6 scorelines by EV (fav-dog, P(exact), EV):")
        for fg, dg, pe, ev in rows[:6]:
            mark = " <-- BEST" if (fg,dg)==(best[0],best[1]) else ""
            print(f"    {fav} {fg}-{dg}: P={pe*100:5.1f}%  EV={ev:.3f}{mark}")
        # best draw scoreline
        draws = [row for row in rows if row[0]==row[1]]
        bd = max(draws, key=lambda x: x[3])
        print(f"  Best PICK : {fav} {best[0]}-{best[1]}  (EV {best[2]:.3f})")
        print(f"  Best DRAW : {bd[0]}-{bd[1]}  (P={bd[2]*100:.1f}%, EV {bd[3]:.3f})")
        print(f"  heatmap -> {path}")
        summary.append({
            "match": f"{fav} v {dog}", "ko": m["ko"],
            "pick": f"{fav} {best[0]}-{best[1]}", "ev": best[2],
            "draw": f"{bd[0]}-{bd[1]}", "draw_ev": bd[3],
        })

    print("\n" + "#"*64)
    print("SUMMARY — best pick per match")
    print(f"{'Match':28} {'KO':22} {'Best pick':18} {'EV':>6}")
    for s in summary:
        print(f"{s['match']:28} {s['ko']:22} {s['pick']:18} {s['ev']:6.3f}")
    print("\nCouldn't price: Germany v Paraguay (R32, KO 23:30 IDT Mon Jun 29) "
          "— kickoff.co.uk page not published yet (404).")
    print("Caveat: bet365 correct-score lists are partial (~15 lines, no "
          "'any other score' bucket); de-vigged within class from listed scorelines only.")


if __name__ == "__main__":
    main()
