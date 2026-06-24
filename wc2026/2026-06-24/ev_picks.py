#!/usr/bin/env python3
"""
World Cup 2026 — bookies-grounded EV correct-score picks + heatmaps.
Date window: 48h forward from 2026-06-24 07:01 IDT.
Source: bet365 odds via kickoff.co.uk (1X2 + correct-score market).

Method:
  - De-vig 1X2 -> P(home), P(draw), P(away)  (implied=1/odds, normalize sum=1).
  - De-vig correct-score WITHIN each outcome class (home-win / draw / away-win),
    normalizing each class to its 1X2 prob. (Listed scorelines only; bet365 omits
    a small "any other score" tail — caveat noted.)
  - All matches are GROUP stage: direction=1pt, exact=3pt.
    EV(scoreline) = P(class) + 2*P(exact).
  - Favorite = team with higher de-vigged win prob (may be the away team).
  - Heatmap axes: favorite goals (y) vs underdog goals (x).
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

# Each match: home, away, 1X2 (home,draw,away) decimal, KO label,
# correct scores as {(home_goals, away_goals): decimal_odds}
MATCHES = [
    dict(home="Switzerland", away="Canada", ko="Wed 24 Jun 22:00 IDT",
         x12=(2.30, 3.20, 3.25),
         cs={(1,1):6.0,(1,0):8.5,(0,1):11.0,(2,1):10.0,(1,2):13.0,(0,0):9.0,
             (2,0):12.0,(0,2):17.0,(2,2):13.0,(3,1):21.0,(1,3):29.0,(3,0):23.0,
             (0,3):34.0,(3,2):29.0,(2,3):41.0}),
    dict(home="Bosnia & H.", away="Qatar", ko="Wed 24 Jun 22:00 IDT",
         x12=(1.40, 5.00, 7.50),
         cs={(2,0):8.0,(2,1):9.0,(1,0):9.0,(1,1):9.5,(3,0):10.0,(3,1):12.0,
             (4,0):17.0,(2,2):19.0,(4,1):19.0,(0,0):17.0,(1,2):23.0,(0,1):23.0,
             (3,2):23.0,(5,0):34.0,(5,1):34.0}),
    dict(home="Scotland", away="Brazil", ko="Thu 25 Jun 01:00 IDT",
         x12=(7.50, 5.25, 1.33),
         cs={(0,2):6.0,(0,1):6.5,(0,3):8.5,(1,2):9.5,(1,1):10.0,(0,0):11.0,
             (1,3):13.0,(0,4):17.0,(1,0):17.0,(1,4):23.0,(2,2):26.0,(2,1):26.0,
             (2,3):34.0,(0,5):34.0,(1,5):41.0}),
    dict(home="Morocco", away="Haiti", ko="Thu 25 Jun 01:00 IDT",
         x12=(1.17, 7.00, 15.00),
         cs={(2,0):6.0,(3,0):7.0,(1,0):8.0,(4,0):10.0,(2,1):11.0,(3,1):12.0,
             (1,1):15.0,(4,1):17.0,(5,0):19.0,(0,0):17.0,(5,1):29.0,(0,1):34.0,
             (2,2):34.0,(6,0):34.0,(3,2):34.0}),
    dict(home="Czechia", away="Mexico", ko="Thu 25 Jun 04:00 IDT",
         x12=(3.60, 3.90, 1.91),
         cs={(0,1):7.0,(1,1):7.5,(0,2):9.0,(1,2):9.0,(0,0):10.0,(1,0):10.0,
             (2,1):13.0,(0,3):17.0,(1,3):17.0,(2,2):19.0,(2,0):21.0,(2,3):34.0,
             (3,1):34.0,(0,4):41.0,(1,4):41.0}),
    dict(home="South Africa", away="South Korea", ko="Thu 25 Jun 04:00 IDT",
         x12=(4.25, 3.40, 1.67),
         cs={(0,1):5.5,(0,2):6.5,(1,1):6.5,(1,2):9.0,(0,0):8.5,(1,0):13.0,
             (0,3):13.0,(1,3):15.0,(2,1):19.0,(2,2):19.0,(2,0):29.0,(0,4):26.0,
             (1,4):29.0,(2,3):29.0,(3,1):41.0}),
    dict(home="Ecuador", away="Germany", ko="Thu 25 Jun 23:00 IDT",
         x12=(2.70, 3.10, 1.90),
         cs={(1,1):8.0,(1,2):9.0,(0,1):8.5,(0,2):10.0,(2,1):13.0,(1,0):13.0,
             (1,3):15.0,(2,2):15.0,(0,0):13.0,(0,3):17.0,(2,0):21.0,(2,3):26.0,
             (3,1):29.0,(1,4):29.0,(3,2):34.0}),
    dict(home="Curacao", away="Cote d'Ivoire", ko="Thu 25 Jun 23:00 IDT",
         x12=(17.0, 8.0, 1.1667),
         cs={(0,2):6.0,(0,3):6.5,(0,1):8.5,(0,4):9.5,(1,2):11.0,(1,3):12.0,
             (1,1):15.0,(0,5):17.0,(1,4):17.0,(0,0):17.0,(1,5):29.0,(0,6):29.0,
             (1,0):34.0,(2,2):34.0,(2,3):41.0}),
    dict(home="Japan", away="Sweden", ko="Fri 26 Jun 02:00 IDT",
         x12=(1.90, 2.40, 3.33),
         cs={(1,1):5.5,(2,1):8.0,(1,0):8.0,(2,0):9.0,(1,2):14.0,(0,1):14.0,
             (3,1):14.0,(2,2):12.0,(0,0):11.0,(3,0):16.0,(0,2):22.0,(3,2):22.0,
             (4,1):28.0,(1,3):33.0,(4,0):33.0}),
    dict(home="Tunisia", away="Netherlands", ko="Fri 26 Jun 02:00 IDT",
         x12=(22.0, 8.0, 1.13),
         cs={(0,2):6.0,(0,3):6.5,(0,4):8.5,(0,1):9.0,(1,2):12.0,(1,3):13.0,
             (0,5):15.0,(1,4):17.0,(1,1):15.0,(0,0):17.0,(0,6):26.0,(1,5):26.0,
             (2,2):41.0,(1,0):41.0,(2,3):41.0}),
    dict(home="Turkiye", away="USA", ko="Fri 26 Jun 05:00 IDT",
         x12=(2.50, 3.10, 1.91),
         cs={(1,1):8.0,(1,2):9.0,(0,1):9.0,(0,2):10.0,(2,1):13.0,(1,0):13.0,
             (2,2):15.0,(1,3):15.0,(0,0):13.0,(0,3):17.0,(2,0):21.0,(2,3):23.0,
             (3,1):29.0,(1,4):34.0,(3,2):34.0}),
    dict(home="Paraguay", away="Australia", ko="Fri 26 Jun 05:00 IDT",
         x12=(2.90, 2.20, 3.45),
         cs={(1,0):5.5,(0,0):4.0,(1,1):5.0,(0,1):8.0,(2,0):10.0,(2,1):11.0,
             (1,2):16.0,(0,2):18.0,(3,0):25.0,(2,2):16.0,(3,1):28.0,(1,3):40.0,
             (0,3):40.0,(3,2):40.0,(4,0):50.0}),
]


def devig_1x2(odds):
    imp = np.array([1.0 / o for o in odds])
    return imp / imp.sum()


def class_of(h, a):
    return "home" if h > a else ("away" if a > h else "draw")


def price_match(m):
    p_home, p_draw, p_away = devig_1x2(m["x12"])
    cls_prob = {"home": p_home, "draw": p_draw, "away": p_away}
    # group scorelines by class, de-vig within class
    buckets = {"home": [], "draw": [], "away": []}
    for (h, a), o in m["cs"].items():
        buckets[class_of(h, a)].append(((h, a), 1.0 / o))
    p_exact = {}
    for cls, items in buckets.items():
        tot = sum(v for _, v in items)
        if tot == 0:
            continue
        for (sc, imp) in items:
            p_exact[sc] = imp / tot * cls_prob[cls]
    # favorite by win prob
    fav_is_home = p_home >= p_away
    fav = m["home"] if fav_is_home else m["away"]
    dog = m["away"] if fav_is_home else m["home"]
    fav_win_p = p_home if fav_is_home else p_away
    dog_win_p = p_away if fav_is_home else p_home
    rows = []
    for (h, a), pe in p_exact.items():
        cls = class_of(h, a)
        pc = cls_prob[cls]
        ev = DIR_PTS * pc + (EXACT_PTS - DIR_PTS) * pe
        # favorite/underdog goal orientation
        fg, dg = (h, a) if fav_is_home else (a, h)
        rows.append(dict(h=h, a=a, fg=fg, dg=dg, cls=cls, pe=pe, ev=ev))
    rows.sort(key=lambda r: r["ev"], reverse=True)
    return dict(m=m, p_home=p_home, p_draw=p_draw, p_away=p_away,
                fav=fav, dog=dog, fav_is_home=fav_is_home,
                fav_win_p=fav_win_p, dog_win_p=dog_win_p,
                cls_prob=cls_prob, rows=rows)


def scoreline_str(r, fav, dog):
    return f"{fav} {r['fg']}-{r['dg']} {dog}"


def heatmap(res, fname):
    m = res["m"]
    fav, dog = res["fav"], res["dog"]
    N = 6  # 0..5
    ev_grid = np.full((N, N), np.nan)
    pe_grid = np.full((N, N), np.nan)
    for r in res["rows"]:
        fg, dg = r["fg"], r["dg"]
        if 0 <= fg < N and 0 <= dg < N:
            ev_grid[fg, dg] = r["ev"]
            pe_grid[fg, dg] = r["pe"]
    best = res["rows"][0]
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    masked = np.ma.masked_invalid(ev_grid)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#f2f2f2")
    vmax = np.nanmax(ev_grid)
    im = ax.imshow(masked, origin="lower", cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    for fg in range(N):
        for dg in range(N):
            if not np.isnan(ev_grid[fg, dg]):
                ev = ev_grid[fg, dg]
                pe = pe_grid[fg, dg] * 100
                txt_col = "white" if ev > 0.55 * vmax else "black"
                ax.text(dg, fg + 0.16, f"{ev:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_col)
                ax.text(dg, fg - 0.24, f"{pe:.1f}%", ha="center", va="center",
                        fontsize=8, color=txt_col)
    # blue box around best EV cell
    ax.add_patch(Rectangle((best["dg"] - 0.5, best["fg"] - 0.5), 1, 1,
                           fill=False, edgecolor="blue", linewidth=3))
    # dotted draw diagonal
    for d in range(N):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                               edgecolor="black", linewidth=1.2, linestyle=(0, (2, 2))))
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xlabel(f"{dog} goals (underdog)", fontsize=11)
    ax.set_ylabel(f"{fav} goals (favorite)", fontsize=11)
    ax.set_title(f"{fav} v {dog} — best: {fav} {best['fg']}-{best['dg']} (EV {best['ev']:.3f})\n"
                 f"{m['ko']}  |  cell=EV, small=P(exact)%", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("EV (points)")
    fig.tight_layout()
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = []
    md = ["# World Cup 2026 — EV correct-score picks (bet365 via kickoff.co.uk)",
          "",
          "Window: 48h from **2026-06-24 07:01 IDT**. All group stage (dir 1pt / exact 3pt; EV = P(class) + 2·P(exact)).",
          "Correct-score de-vigged within outcome class from listed scorelines only (bet365 omits a small 'any other score' tail — minor caveat).",
          ""]
    for m in MATCHES:
        res = price_match(m)
        fav, dog = res["fav"], res["dog"]
        slug = (fav + "_v_" + dog).lower().replace(" ", "_").replace("&", "and").replace("'", "").replace(".", "")
        fname = os.path.join(OUTDIR, f"{slug}.png")
        heatmap(res, fname)
        best = res["rows"][0]
        # best draw scoreline
        draws = [r for r in res["rows"] if r["cls"] == "draw"]
        best_draw = draws[0] if draws else None
        summary.append(dict(match=f"{m['home']} v {m['away']}", ko=m["ko"],
                            fav=fav, dog=dog, pick=f"{fav} {best['fg']}-{best['dg']}",
                            ev=best["ev"], pe=best["pe"]))
        md.append(f"## {m['home']} v {m['away']} — {m['ko']}")
        md.append(f"De-vig 1X2: {m['home']} {res['p_home']*100:.1f}% / Draw {res['p_draw']*100:.1f}% / {m['away']} {res['p_away']*100:.1f}%  "
                  f"→ **favorite: {fav}** ({res['fav_win_p']*100:.1f}%)")
        md.append("")
        md.append("| # | Scoreline (Fav-Dog) | P(exact) | EV |")
        md.append("|---|---|---|---|")
        for i, r in enumerate(res["rows"][:6], 1):
            md.append(f"| {i} | {fav} {r['fg']}-{r['dg']} | {r['pe']*100:.1f}% | {r['ev']:.3f} |")
        md.append("")
        md.append(f"**Best pick: {fav} {best['fg']}-{best['dg']}** (EV {best['ev']:.3f}, P {best['pe']*100:.1f}%). ")
        if best_draw:
            md.append(f"Best DRAW for contrast: {best_draw['fg']}-{best_draw['dg']} (EV {best_draw['ev']:.3f}, P {best_draw['pe']*100:.1f}%).")
        md.append(f"\n![{fav} v {dog}]({slug}.png)\n")
        print(f"{m['home']} v {m['away']}: pick {fav} {best['fg']}-{best['dg']} EV {best['ev']:.3f}")

    md.append("## Summary — best pick per match")
    md.append("")
    md.append("| Match | KO (IDT) | Favorite | Best pick | EV | P(exact) |")
    md.append("|---|---|---|---|---|---|")
    for s in sorted(summary, key=lambda x: x["ev"], reverse=True):
        md.append(f"| {s['match']} | {s['ko'].replace(' IDT','')} | {s['fav']} | {s['pick']} | {s['ev']:.3f} | {s['pe']*100:.1f}% |")
    md.append("")
    with open(os.path.join(OUTDIR, "README.md"), "w") as f:
        f.write("\n".join(md))
    print("\nWrote README.md and", len(MATCHES), "heatmaps to", OUTDIR)


if __name__ == "__main__":
    main()
