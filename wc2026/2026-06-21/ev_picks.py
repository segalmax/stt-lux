#!/usr/bin/env python3
"""WC2026 bookies-grounded EV picks — 2026-06-21 (Asia/Jerusalem).

Only one upcoming match today after 21:59 IDT: Belgium v Iran (KO 22:00 IDT / 3pm ET).
Group stage. bet365 odds via kickoff.co.uk.

Scoring (group stage): direction 1 pt, exact 3 pts.
  EV = P(outcome class) + 2 * P(exact)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTDIR = os.path.dirname(os.path.abspath(__file__))

def frac(n, d=1):
    """Fractional odds -> decimal."""
    return n / d + 1.0

# ---- bet365 1X2 (fractional) ----
# Belgium 2/5, Draw 15/4, Iran 7/1
o_home = frac(2, 5)   # Belgium win
o_draw = frac(15, 4)  # Draw
o_away = frac(7, 1)   # Iran win

# de-vig 1X2
imp = np.array([1/o_home, 1/o_draw, 1/o_away])
p_class = imp / imp.sum()
P_HOME, P_DRAW, P_AWAY = p_class
print(f"1X2 de-vigged: Belgium {P_HOME:.3f}  Draw {P_DRAW:.3f}  Iran {P_AWAY:.3f}")

# favorite = higher win prob
if P_HOME >= P_AWAY:
    FAV, DOG = "Belgium", "Iran"
    P_FAVWIN, P_DOGWIN = P_HOME, P_AWAY
    fav_is_home = True
else:
    FAV, DOG = "Iran", "Belgium"
    P_FAVWIN, P_DOGWIN = P_AWAY, P_HOME
    fav_is_home = False
print(f"Favorite: {FAV} (win {P_FAVWIN:.3f})  Underdog: {DOG} (win {P_DOGWIN:.3f})")

# ---- correct score market (home-away orientation: Belgium home, Iran away) ----
# (belgium_goals, iran_goals): decimal odds
cs = {
    (2, 0): frac(6),    (1, 0): frac(6),    (2, 1): frac(8),
    (1, 1): frac(8),    (3, 0): frac(17, 2),(3, 1): frac(12),
    (0, 0): frac(11),   (4, 0): frac(18),   (0, 1): frac(18),   # Iran 1-0 = belgium 0, iran 1
    (2, 2): frac(20),   (4, 1): frac(22),   (1, 2): frac(22),   # Iran 2-1 = belgium 1, iran 2
    (3, 2): frac(28),   (5, 0): frac(33),   (0, 2): frac(40),   # Iran 2-0 = belgium 0, iran 2
}

# classify each scoreline & implied prob
home_imp, draw_imp, away_imp = {}, {}, {}
for (bg, ig), od in cs.items():
    ip = 1.0 / od
    if bg > ig:
        home_imp[(bg, ig)] = ip
    elif bg == ig:
        draw_imp[(bg, ig)] = ip
    else:
        away_imp[(bg, ig)] = ip

# de-vig within class -> each class sums to its 1X2 prob
def normalize_class(d, target):
    s = sum(d.values())
    return {k: v / s * target for k, v in d.items()}

p_exact = {}
p_exact.update(normalize_class(home_imp, P_HOME))
p_exact.update(normalize_class(draw_imp, P_DRAW))
p_exact.update(normalize_class(away_imp, P_AWAY))

# ---- EV per scoreline (group stage) ----
def outcome_class_prob(bg, ig):
    if bg > ig: return P_HOME
    if bg == ig: return P_DRAW
    return P_AWAY

DIR_PTS, EXACT_PTS = 1, 3  # group stage
rows = []
for (bg, ig), pe in p_exact.items():
    pcls = outcome_class_prob(bg, ig)
    ev = DIR_PTS * pcls + (EXACT_PTS - DIR_PTS) * pe
    rows.append(((bg, ig), pe, ev))

rows.sort(key=lambda r: -r[2])

# map to favorite/underdog axes for display
def fav_dog(bg, ig):
    """Return (fav_goals, dog_goals)."""
    return (bg, ig) if fav_is_home else (ig, bg)

print("\nTop scorelines by EV (Belgium-Iran):")
for (bg, ig), pe, ev in rows[:6]:
    print(f"  Bel {bg}-{ig} Iran | P(exact) {pe*100:5.2f}%  EV {ev:.3f}")

best = rows[0]
best_draw = max((r for r in rows if r[0][0] == r[0][1]), key=lambda r: r[2])
print(f"\nBEST PICK: Bel {best[0][0]}-{best[0][1]} Iran  EV {best[2]:.3f}  P {best[1]*100:.2f}%")
print(f"BEST DRAW: {best_draw[0][0]}-{best_draw[0][1]}  EV {best_draw[2]:.3f}  P {best_draw[1]*100:.2f}%")

# ---- HEATMAP: favorite goals (y) vs underdog goals (x), 0..5 ----
N = 6
ev_grid = np.full((N, N), np.nan)
pe_grid = np.zeros((N, N))
for (bg, ig), pe, ev in rows:
    fg, dg = fav_dog(bg, ig)
    if 0 <= fg < N and 0 <= dg < N:
        ev_grid[fg, dg] = ev
        pe_grid[fg, dg] = pe

# fill unlisted cells with 0 EV for coloring context
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
            ax.text(dg, fg - 0.22, f"{pe*100:.1f}%", ha="center", va="center",
                    fontsize=8, color="black")
        else:
            ax.text(dg, fg, "·", ha="center", va="center", fontsize=10, color="gray")

# best-EV cell -> blue box (in fav/dog coords)
bfg, bdg = fav_dog(*best[0])
ax.add_patch(Rectangle((bdg - 0.5, bfg - 0.5), 1, 1, fill=False, edgecolor="blue", lw=3))

# draw diagonal -> dotted outline
for k in range(N):
    ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                            edgecolor="black", lw=1.2, linestyle=":"))

ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xlabel(f"{DOG} goals", fontsize=11)
ax.set_ylabel(f"{FAV} goals", fontsize=11)
ax.set_title(f"{FAV} v {DOG} — best: {FAV} {bfg}-{bdg} (EV {best[2]:.2f})\n"
             f"group stage · bet365 de-vigged · KO 22:00 IDT", fontsize=12)
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("EV (points)")
fig.tight_layout()
outpath = os.path.join(OUTDIR, "belgium_v_iran_heatmap.png")
fig.savefig(outpath, dpi=140)
print(f"\nSaved {outpath}")

# ---- write summary markdown ----
md = []
md.append("# WC2026 EV picks — 2026-06-21 (IDT)\n")
md.append("Only one upcoming match after 21:59 IDT. Spain–Saudi already played (19:00 IDT); "
          "Uruguay–Cape Verde (01:00) & NZ–Egypt (04:00) roll into Jun 22 → excluded.\n")
md.append("**Caveat:** bet365 correct-score list is partial (15 lines, no 'any other score' "
          "bucket); de-vigged within each outcome class from scorelines listed.\n")
md.append("## Belgium v Iran — Group stage · KO 22:00 IDT (3pm ET)\n")
md.append(f"De-vigged 1X2: Belgium **{P_HOME*100:.1f}%** · Draw **{P_DRAW*100:.1f}%** · "
          f"Iran **{P_AWAY*100:.1f}%**. Favorite: **{FAV}**.\n")
md.append("### Top 6 scorelines by EV\n")
md.append("| Scoreline (Bel-Iran) | P(exact) | EV |")
md.append("|---|---|---|")
for (bg, ig), pe, ev in rows[:6]:
    md.append(f"| {bg}-{ig} | {pe*100:.2f}% | {ev:.3f} |")
md.append("")
md.append(f"**Best pick:** Belgium {best[0][0]}-{best[0][1]} — EV {best[2]:.3f} "
          f"(P {best[1]*100:.2f}%)  \n")
md.append(f"**Best draw (contrast):** {best_draw[0][0]}-{best_draw[0][1]} — EV "
          f"{best_draw[2]:.3f} (P {best_draw[1]*100:.2f}%)\n")
md.append("## Summary across matches\n")
md.append("| Match | Stage | KO IDT | Best pick | EV |")
md.append("|---|---|---|---|---|")
md.append(f"| Belgium v Iran | Group | 22:00 | Bel {best[0][0]}-{best[0][1]} | {best[2]:.3f} |")
md.append("")
with open(os.path.join(OUTDIR, "summary.md"), "w") as f:
    f.write("\n".join(md))
print("Saved summary.md")
