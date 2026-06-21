#!/usr/bin/env python3
"""WC2026 bookies-grounded EV picks — 48h window from 2026-06-21 22:13 IDT.

Upcoming matches in the next 48h (kickoffs in IDT):
  1. Uruguay v Cape Verde      01:00 IDT Jun 22  (6pm ET Jun 21)
  2. New Zealand v Egypt       04:00 IDT Jun 22  (9pm ET Jun 21)
  3. Argentina v Austria       20:00 IDT Jun 22  (1pm ET Jun 22)
  4. France v Iraq             00:00 IDT Jun 23  (5pm ET Jun 22)
  5. Norway v Senegal          03:00 IDT Jun 23  (8pm ET Jun 22)
  6. Jordan v Algeria          06:00 IDT Jun 23  (11pm ET Jun 22)
  7. Portugal v Uzbekistan     20:00 IDT Jun 23  (1pm ET Jun 23)

All group stage: direction 1 pt, exact 3 pts -> EV = P(class) + 2*P(exact).
bet365 odds via kickoff.co.uk. Belgium v Iran (22:00 IDT Jun 21) already kicked
off -> excluded. England v Ghana (23:00 IDT Jun 23) is just outside the 48h window.

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
DIR_PTS, EXACT_PTS = 1, 3  # group stage


def F(n, d=1):
    """Fractional odds -> decimal."""
    return n / d + 1.0


# Each match: home, away, KO (IDT), 1X2 decimal odds, correct-score {(hg,ag): decimal}
MATCHES = [
    dict(
        slug="uruguay_v_cape_verde", home="Uruguay", away="Cape Verde",
        ko="01:00 IDT Jun 22",
        o1x2=(F(21, 50), F(10, 3), F(15, 2)),
        cs={(1, 0): F(4), (2, 0): F(9, 2), (1, 1): F(8), (3, 0): F(8),
            (0, 0): F(7), (2, 1): F(9), (3, 1): F(14), (0, 1): F(16),
            (4, 0): F(18), (1, 2): F(28), (2, 2): F(28), (4, 1): F(28),
            (5, 0): F(40), (3, 2): F(40), (0, 2): F(40)},
    ),
    dict(
        slug="new_zealand_v_egypt", home="New Zealand", away="Egypt",
        ko="04:00 IDT Jun 22",
        o1x2=(F(19, 4), F(31, 10), F(4, 7)),
        cs={(0, 1): F(11, 2), (0, 2): F(6), (1, 1): F(7), (1, 2): F(8),
            (0, 0): F(17, 2), (0, 3): F(11), (1, 0): F(14), (1, 3): F(14),
            (2, 1): F(20), (2, 2): F(20), (0, 4): F(22), (1, 4): F(28),
            (2, 0): F(33), (2, 3): F(33), (3, 1): F(50)},
    ),
    dict(
        slug="argentina_v_austria", home="Argentina", away="Austria",
        ko="20:00 IDT Jun 22",
        o1x2=(F(8, 15), F(16, 5), F(11, 2)),
        cs={(1, 0): F(11, 2), (2, 0): F(6), (2, 1): F(8), (3, 0): F(10),
            (3, 1): F(12), (4, 0): F(20), (4, 1): F(25), (3, 2): F(28),
            (5, 0): F(40), (0, 0): F(9), (1, 1): F(7), (2, 2): F(20),
            (0, 1): F(16), (0, 2): F(33), (1, 2): F(20)},
    ),
    dict(
        slug="france_v_iraq", home="France", away="Iraq",
        ko="00:00 IDT Jun 23",
        o1x2=(1.07, 11.00, 34.00),
        cs={(3, 0): 6.00, (2, 0): 6.50, (4, 0): 7.00, (1, 0): 10.00,
            (5, 0): 11.00, (3, 1): 15.00, (2, 1): 15.00, (4, 1): 17.00,
            (6, 0): 19.00, (1, 1): 21.00, (5, 1): 26.00, (0, 0): 21.00,
            (7, 0): 34.00, (6, 1): 41.00, (3, 2): 51.00},
    ),
    dict(
        slug="norway_v_senegal", home="Norway", away="Senegal",
        ko="03:00 IDT Jun 23",
        o1x2=(2.25, 3.60, 3.00),
        cs={(1, 1): 7.00, (1, 0): 8.50, (2, 1): 9.50, (0, 1): 10.00,
            (1, 2): 12.00, (2, 0): 12.00, (0, 0): 11.00, (2, 2): 15.00,
            (0, 2): 17.00, (3, 1): 19.00, (3, 0): 23.00, (1, 3): 26.00,
            (3, 2): 29.00, (2, 3): 34.00, (0, 3): 34.00},
    ),
    dict(
        slug="jordan_v_algeria", home="Jordan", away="Algeria",
        ko="06:00 IDT Jun 23",
        o1x2=(F(9, 2), F(16, 5), F(4, 7)),
        cs={(0, 1): F(6), (0, 2): F(13, 2), (1, 1): F(7), (1, 2): F(8),
            (0, 3): F(11), (0, 0): F(10), (1, 3): F(14), (1, 0): F(14),
            (2, 1): F(18), (2, 2): F(18), (0, 4): F(22), (1, 4): F(25),
            (2, 3): F(28), (2, 0): F(28), (3, 1): F(50)},
    ),
    dict(
        slug="portugal_v_uzbekistan", home="Portugal", away="Uzbekistan",
        ko="20:00 IDT Jun 23",
        o1x2=(1.20, 7.00, 13.00),
        cs={(2, 0): 6.00, (3, 0): 7.00, (1, 0): 8.00, (2, 1): 11.00,
            (4, 0): 11.00, (3, 1): 12.00, (1, 1): 13.00, (0, 0): 15.00,
            (4, 1): 19.00, (5, 0): 21.00, (0, 1): 26.00, (2, 2): 34.00,
            (5, 1): 34.00, (3, 2): 34.00, (1, 2): 34.00},
    ),
]


def normalize_class(d, target):
    s = sum(d.values())
    if s == 0:
        return {}
    return {k: v / s * target for k, v in d.items()}


def process(m):
    home, away = m["home"], m["away"]
    o_home, o_draw, o_away = m["o1x2"]

    # de-vig 1X2
    imp = np.array([1 / o_home, 1 / o_draw, 1 / o_away])
    P_HOME, P_DRAW, P_AWAY = imp / imp.sum()

    # favorite = higher win prob (may be the away team)
    if P_HOME >= P_AWAY:
        FAV, DOG, fav_is_home = home, away, True
        P_FAVWIN = P_HOME
    else:
        FAV, DOG, fav_is_home = away, home, False
        P_FAVWIN = P_AWAY

    # classify correct-score lines & de-vig within class
    home_imp, draw_imp, away_imp = {}, {}, {}
    for (hg, ag), od in m["cs"].items():
        ip = 1.0 / od
        if hg > ag:
            home_imp[(hg, ag)] = ip
        elif hg == ag:
            draw_imp[(hg, ag)] = ip
        else:
            away_imp[(hg, ag)] = ip

    p_exact = {}
    p_exact.update(normalize_class(home_imp, P_HOME))
    p_exact.update(normalize_class(draw_imp, P_DRAW))
    p_exact.update(normalize_class(away_imp, P_AWAY))

    def class_prob(hg, ag):
        return P_HOME if hg > ag else (P_DRAW if hg == ag else P_AWAY)

    rows = []
    for (hg, ag), pe in p_exact.items():
        pcls = class_prob(hg, ag)
        ev = DIR_PTS * pcls + (EXACT_PTS - DIR_PTS) * pe
        rows.append(((hg, ag), pe, ev))
    rows.sort(key=lambda r: -r[2])

    def fav_dog(hg, ag):
        return (hg, ag) if fav_is_home else (ag, hg)

    best = rows[0]
    draws = [r for r in rows if r[0][0] == r[0][1]]
    best_draw = max(draws, key=lambda r: r[2]) if draws else None

    # ---- HEATMAP: favorite goals (y) vs underdog goals (x), 0..5 ----
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
            pe = pe_grid[fg, dg]
            if not np.isnan(ev):
                ax.text(dg, fg + 0.16, f"{ev:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="black")
                ax.text(dg, fg - 0.22, f"{pe*100:.1f}%", ha="center",
                        va="center", fontsize=8, color="black")
            else:
                ax.text(dg, fg, "·", ha="center", va="center",
                        fontsize=10, color="gray")

    bfg, bdg = fav_dog(*best[0])
    ax.add_patch(Rectangle((bdg - 0.5, bfg - 0.5), 1, 1, fill=False,
                           edgecolor="blue", lw=3))
    for k in range(N):
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=1.2, linestyle=":"))

    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xlabel(f"{DOG} goals", fontsize=11)
    ax.set_ylabel(f"{FAV} goals", fontsize=11)
    ax.set_title(f"{FAV} v {DOG} — best: {FAV} {bfg}-{bdg} (EV {best[2]:.2f})\n"
                 f"group stage · bet365 de-vigged · KO {m['ko']}", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("EV (points)")
    fig.tight_layout()
    outpath = os.path.join(OUTDIR, f"{m['slug']}_heatmap.png")
    fig.savefig(outpath, dpi=140)
    plt.close(fig)

    return dict(home=home, away=away, FAV=FAV, DOG=DOG, ko=m["ko"],
                P_HOME=P_HOME, P_DRAW=P_DRAW, P_AWAY=P_AWAY,
                rows=rows, best=best, best_draw=best_draw,
                bfg=bfg, bdg=bdg, fav_is_home=fav_is_home, outpath=outpath)


results = []
for m in MATCHES:
    r = process(m)
    results.append(r)
    print(f"\n=== {r['home']} v {r['away']} — KO {r['ko']} ===")
    print(f"1X2 de-vig: {r['home']} {r['P_HOME']*100:.1f}%  Draw "
          f"{r['P_DRAW']*100:.1f}%  {r['away']} {r['P_AWAY']*100:.1f}%  "
          f"| FAV={r['FAV']}")
    print("Top 6 by EV (home-away):")
    for (hg, ag), pe, ev in r["rows"][:6]:
        print(f"  {r['home']} {hg}-{ag} {r['away']} | P {pe*100:5.2f}%  EV {ev:.3f}")
    b = r["best"]
    print(f"BEST: {r['home']} {b[0][0]}-{b[0][1]} {r['away']} "
          f"(= {r['FAV']} {r['bfg']}-{r['bdg']}) EV {b[2]:.3f}")
    if r["best_draw"]:
        bd = r["best_draw"]
        print(f"BEST DRAW: {bd[0][0]}-{bd[0][1]} EV {bd[2]:.3f}")
    print(f"Saved {r['outpath']}")

# ---- summary markdown ----
md = ["# WC2026 EV picks — 48h window from 2026-06-21 22:13 IDT\n"]
md.append("All group stage. bet365 odds via kickoff.co.uk, de-vigged. "
          "**Caveat:** correct-score list is partial (~15 lines, no 'any other "
          "score' bucket); de-vigged within each outcome class from listed scorelines.\n")
md.append("Belgium v Iran (22:00 IDT Jun 21) already kicked off → excluded. "
          "England v Ghana (23:00 IDT Jun 23) is just outside the 48h window.\n")

for r in results:
    b = r["best"]
    md.append(f"## {r['home']} v {r['away']} — KO {r['ko']}\n")
    md.append(f"De-vig 1X2: {r['home']} **{r['P_HOME']*100:.1f}%** · Draw "
              f"**{r['P_DRAW']*100:.1f}%** · {r['away']} **{r['P_AWAY']*100:.1f}%**. "
              f"Favorite: **{r['FAV']}**.\n")
    md.append("| Scoreline (home-away) | P(exact) | EV |")
    md.append("|---|---|---|")
    for (hg, ag), pe, ev in r["rows"][:6]:
        md.append(f"| {r['home']} {hg}-{ag} {r['away']} | {pe*100:.2f}% | {ev:.3f} |")
    md.append(f"\n**Best pick:** {r['home']} {b[0][0]}-{b[0][1]} {r['away']} "
              f"(= {r['FAV']} {r['bfg']}-{r['bdg']}) — EV {b[2]:.3f} (P {b[1]*100:.2f}%)  ")
    if r["best_draw"]:
        bd = r["best_draw"]
        md.append(f"**Best draw (contrast):** {bd[0][0]}-{bd[0][1]} — "
                  f"EV {bd[2]:.3f} (P {bd[1]*100:.2f}%)\n")

md.append("## Summary across all matches\n")
md.append("| Match | KO IDT | Favorite | Best pick | EV |")
md.append("|---|---|---|---|---|")
for r in results:
    b = r["best"]
    md.append(f"| {r['home']} v {r['away']} | {r['ko']} | {r['FAV']} | "
              f"{r['FAV']} {r['bfg']}-{r['bdg']} | {b[2]:.3f} |")
md.append("")
with open(os.path.join(OUTDIR, "summary.md"), "w") as f:
    f.write("\n".join(md))
print("\nSaved summary.md")
