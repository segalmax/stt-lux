#!/usr/bin/env python3
"""WC2026 bookies-grounded EV picks — 48h window from 2026-07-01 07:01 IDT.

Round of 32 matches kicking off in the next 48h (kickoffs in IDT; ET+7):
   1. England v Congo DR       19:00 IDT Wed Jul 1  (12pm ET)
   2. Belgium v Senegal        23:00 IDT Wed Jul 1  (4pm ET)
   3. USA v Bosnia & Herz.     03:00 IDT Thu Jul 2  (8pm ET Jul 1)
   4. Spain v Austria          22:00 IDT Thu Jul 2  (3pm ET)
   5. Portugal v Croatia       02:00 IDT Fri Jul 3  (7pm ET Jul 2)
   6. Switzerland v Algeria    06:00 IDT Fri Jul 3  (11pm ET Jul 2)

Excluded: Mexico v Ecuador (9pm ET Jun 30 -> 04:00 IDT Jul 1) already played.
Jul 3 games (Australia/Argentina/Colombia, 2pm+ ET -> 21:00 IDT Jul 3 onward)
fall beyond the 48h window (ends 07:01 IDT Fri Jul 3) -> excluded.

All Round of 32: direction 2 pts, exact 5 pts -> EV = 2*P(class) + 3*P(exact).
bet365 odds via kickoff.co.uk, de-vigged.

Caveat: bet365 correct-score list from the page is partial (~15 lines, no
"any other score" bucket); de-vigged WITHIN each outcome class from the
scorelines actually listed.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTDIR = os.path.dirname(os.path.abspath(__file__))

def f(n, d=1):
    """Fractional odds -> decimal."""
    return n / d + 1.0

# ---------------------------------------------------------------------------
# MATCH DATA — all correct-score dicts are (home_goals, away_goals): decimal odds
# ---------------------------------------------------------------------------
MATCHES = [
    {
        "home": "England", "away": "Congo DR", "stage": "r32",
        "ko": "19:00 IDT Wed Jul 1",
        "x12": {"home": f(2,7), "draw": f(17,4), "away": f(12,1)},
        "cs": {
            (2,0):f(4,1),(1,0):f(9,2),(3,0):f(13,2),(2,1):f(10,1),(4,0):f(12,1),
            (3,1):f(14,1),(5,0):f(28,1),(3,2):f(40,1),(5,1):f(40,1),
            (1,1):f(10,1),(0,0):f(8,1),(2,2):f(33,1),
            (0,1):f(20,1),(1,2):f(40,1),
        },
    },
    {
        "home": "Belgium", "away": "Senegal", "stage": "r32",
        "ko": "23:00 IDT Wed Jul 1",
        "x12": {"home": f(23,20), "draw": f(9,4), "away": f(5,2)},
        "cs": {
            (1,0):f(7,1),(2,1):f(17,2),(2,0):f(10,1),(3,1):f(18,1),(3,0):f(20,1),(3,2):f(28,1),
            (1,1):f(11,2),(0,0):f(8,1),(2,2):f(14,1),
            (0,1):f(10,1),(1,2):f(12,1),(0,2):f(18,1),(1,3):f(28,1),(2,3):f(40,1),(0,3):f(40,1),
        },
    },
    {
        "home": "USA", "away": "Bosnia & Herz.", "stage": "r32",
        "ko": "03:00 IDT Thu Jul 2",
        "x12": {"home": f(2,5), "draw": f(4,1), "away": f(13,2)},
        "cs": {
            (2,0):f(6,1),(1,0):f(6,1),(2,1):f(8,1),(3,0):f(17,2),(3,1):f(12,1),
            (4,0):f(16,1),(4,1):f(20,1),(3,2):f(28,1),(5,0):f(33,1),
            (1,1):f(17,2),(0,0):f(12,1),(2,2):f(22,1),
            (0,1):f(18,1),(1,2):f(22,1),(0,2):f(40,1),
        },
    },
    {
        "home": "Spain", "away": "Austria", "stage": "r32",
        "ko": "22:00 IDT Thu Jul 2",
        "x12": {"home": 1.33, "draw": 5.25, "away": 8.50},
        "cs": {
            (2,0):6.00,(1,0):6.50,(3,0):8.50,(2,1):9.50,(3,1):13.00,(4,0):15.00,
            (4,1):23.00,(5,0):29.00,(3,2):34.00,(5,1):41.00,
            (1,1):10.00,(0,0):11.00,(2,2):26.00,
            (0,1):21.00,(1,2):29.00,
        },
    },
    {
        "home": "Portugal", "away": "Croatia", "stage": "r32",
        "ko": "02:00 IDT Fri Jul 3",
        "x12": {"home": 1.80, "draw": 3.50, "away": 4.75},
        "cs": {
            (1,0):7.00,(2,0):8.50,(2,1):9.00,(3,0):15.00,(3,1):17.00,(3,2):34.00,(4,0):34.00,(4,1):34.00,
            (1,1):6.50,(0,0):8.50,(2,2):17.00,
            (0,1):13.00,(1,2):17.00,(0,2):26.00,(1,3):41.00,
        },
    },
    {
        "home": "Switzerland", "away": "Algeria", "stage": "r32",
        "ko": "06:00 IDT Fri Jul 3",
        "x12": {"home": f(1,1), "draw": f(23,10), "away": f(3,1)},
        "cs": {
            (1,0):f(6,1),(2,0):f(17,2),(2,1):f(17,2),(3,0):f(18,1),(3,1):f(18,1),(3,2):f(33,1),(4,0):f(40,1),(4,1):f(40,1),
            (1,1):f(11,2),(0,0):f(15,2),(2,2):f(16,1),
            (0,1):f(10,1),(1,2):f(14,1),(0,2):f(20,1),(1,3):f(33,1),
        },
    },
]

STAGE_PTS = {  # (direction, exact)
    "group": (1, 3), "r32": (2, 5), "r16": (2, 5),
    "qf": (4, 8), "sf": (5, 10), "third": (5, 10), "final": (8, 15),
}

def slug(s):
    return s.lower().replace(" ", "_").replace("&", "and").replace(".", "")

def process(m):
    o = m["x12"]
    imp = np.array([1/o["home"], 1/o["draw"], 1/o["away"]])
    P_HOME, P_DRAW, P_AWAY = imp / imp.sum()

    # classify correct-score lines & implied probs
    home_imp, draw_imp, away_imp = {}, {}, {}
    for (hg, ag), od in m["cs"].items():
        ip = 1.0 / od
        if hg > ag:   home_imp[(hg, ag)] = ip
        elif hg == ag: draw_imp[(hg, ag)] = ip
        else:          away_imp[(hg, ag)] = ip

    def norm(d, target):
        s = sum(d.values())
        return {k: v / s * target for k, v in d.items()} if s > 0 else {}

    p_exact = {}
    p_exact.update(norm(home_imp, P_HOME))
    p_exact.update(norm(draw_imp, P_DRAW))
    p_exact.update(norm(away_imp, P_AWAY))

    # favorite by win prob (may be the away team)
    if P_HOME >= P_AWAY:
        fav, dog = m["home"], m["away"]
        p_favwin, p_dogwin = P_HOME, P_AWAY
        fav_is_home = True
    else:
        fav, dog = m["away"], m["home"]
        p_favwin, p_dogwin = P_AWAY, P_HOME
        fav_is_home = False

    dir_pts, exact_pts = STAGE_PTS[m["stage"]]

    def cls_prob(hg, ag):
        if hg > ag:  return P_HOME
        if hg == ag: return P_DRAW
        return P_AWAY

    rows = []
    for (hg, ag), pe in p_exact.items():
        ev = dir_pts * cls_prob(hg, ag) + (exact_pts - dir_pts) * pe
        rows.append(((hg, ag), pe, ev))
    rows.sort(key=lambda r: -r[2])

    def fav_dog(hg, ag):
        """(home,away) -> (fav_goals, dog_goals)."""
        return (hg, ag) if fav_is_home else (ag, hg)

    return dict(m=m, P_HOME=P_HOME, P_DRAW=P_DRAW, P_AWAY=P_AWAY,
                fav=fav, dog=dog, p_favwin=p_favwin, p_dogwin=p_dogwin,
                fav_is_home=fav_is_home, fav_dog=fav_dog, rows=rows,
                dir_pts=dir_pts, exact_pts=exact_pts)

def heatmap(r):
    m, rows, fav_dog = r["m"], r["rows"], r["fav_dog"]
    fav, dog = r["fav"], r["dog"]
    best = rows[0]
    N = 6
    ev_grid = np.full((N, N), np.nan)
    pe_grid = np.zeros((N, N))
    for (hg, ag), pe, ev in rows:
        fg, dg = fav_dog(hg, ag)
        if 0 <= fg < N and 0 <= dg < N:
            ev_grid[fg, dg] = ev
            pe_grid[fg, dg] = pe
    ev_plot = np.nan_to_num(ev_grid, nan=0.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(ev_plot, origin="lower", cmap="YlOrRd", aspect="equal")
    for fg in range(N):
        for dg in range(N):
            ev = ev_grid[fg, dg]
            if not np.isnan(ev):
                ax.text(dg, fg + 0.16, f"{ev:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="black")
                ax.text(dg, fg - 0.24, f"{pe_grid[fg, dg]*100:.1f}%", ha="center",
                        va="center", fontsize=8, color="black")
            else:
                ax.text(dg, fg, "·", ha="center", va="center", fontsize=10, color="gray")

    bfg, bdg = fav_dog(*best[0])
    ax.add_patch(Rectangle((bdg - 0.5, bfg - 0.5), 1, 1, fill=False, edgecolor="blue", lw=3))
    for k in range(N):  # draw diagonal dotted
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=1.2, linestyle=":"))
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xlabel(f"{dog} goals", fontsize=11)
    ax.set_ylabel(f"{fav} goals", fontsize=11)
    ax.set_title(f"{fav} v {dog} — best: {fav} {bfg}-{bdg} (EV {best[2]:.2f})\n"
                 f"{m['stage'].upper()} · bet365 de-vigged · KO {m['ko']}", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("EV (points)")
    fig.tight_layout()
    fname = f"{slug(fav)}_v_{slug(dog)}_heatmap.png"
    fig.savefig(os.path.join(OUTDIR, fname), dpi=140)
    plt.close(fig)
    return fname

def main():
    md = ["# WC2026 EV picks — 2026-07-01 (IDT, 48h window)\n",
          "6 upcoming Round-of-32 matches in the next 48h (kickoffs Wed Jul 1 19:00 IDT "
          "→ Fri Jul 3 06:00 IDT). bet365 odds via kickoff.co.uk, de-vigged.\n",
          "**Scoring (R32):** direction 2 pts, exact 5 pts → EV = 2·P(class) + 3·P(exact).\n",
          "**Caveat:** bet365 correct-score list is partial (~15 lines, no 'any other score' "
          "bucket); de-vigged WITHIN each outcome class from scorelines listed.\n"]
    summary = []
    print(f"{'MATCH':<28}{'KO IDT':<22}{'BEST PICK':<22}{'EV':>6}")
    for m in MATCHES:
        r = process(m)
        rows = r["rows"]
        fav, dog = r["fav"], r["dog"]
        best = rows[0]
        bfg, bdg = r["fav_dog"](*best[0])
        draws = [x for x in rows if x[0][0] == x[0][1]]
        best_draw = max(draws, key=lambda x: x[2]) if draws else None
        fname = heatmap(r)

        label = f"{m['home']} v {m['away']}"
        pick = f"{fav} {bfg}-{bdg}"
        print(f"{label:<28}{m['ko']:<22}{pick:<22}{best[2]:>6.3f}")

        summary.append((label, m["ko"], pick, best[2], best_draw))

        md.append(f"## {label} — {m['stage'].upper()} · KO {m['ko']}\n")
        md.append(f"De-vigged 1X2: {m['home']} **{r['P_HOME']*100:.1f}%** · "
                  f"Draw **{r['P_DRAW']*100:.1f}%** · {m['away']} **{r['P_AWAY']*100:.1f}%**. "
                  f"Favorite: **{fav}** (win {r['p_favwin']*100:.1f}%).\n")
        md.append("| Scoreline (fav-dog) | P(exact) | EV |")
        md.append("|---|---|---|")
        for (hg, ag), pe, ev in rows[:6]:
            fg, dg = r["fav_dog"](hg, ag)
            md.append(f"| {fav} {fg}-{dg} | {pe*100:.2f}% | {ev:.3f} |")
        md.append("")
        md.append(f"**Best pick:** {fav} {bfg}-{bdg} — EV {best[2]:.3f} "
                  f"(P {best[1]*100:.2f}%)  ")
        if best_draw:
            md.append(f"**Best draw (contrast):** {best_draw[0][0]}-{best_draw[0][1]} — "
                      f"EV {best_draw[2]:.3f} (P {best_draw[1]*100:.2f}%)  ")
        md.append(f"![heatmap]({fname})\n")

    md.append("## Summary across all matches\n")
    md.append("| Match | KO IDT | Best pick | EV | Best draw |")
    md.append("|---|---|---|---|---|")
    for label, ko, pick, ev, bd in summary:
        bdtxt = f"{bd[0][0]}-{bd[0][1]} (EV {bd[2]:.2f})" if bd else "—"
        md.append(f"| {label} | {ko} | {pick} | {ev:.3f} | {bdtxt} |")
    md.append("")
    with open(os.path.join(OUTDIR, "summary.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\nSaved summary.md and heatmaps to", OUTDIR)

if __name__ == "__main__":
    main()
