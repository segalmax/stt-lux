#!/usr/bin/env python3
"""World Cup 2026 bookies-grounded EV picks + heatmaps.

Data source: bet365 odds via kickoff.co.uk (1X2 + correct-score market).
De-vig 1X2 -> P(home/draw/away). De-vig correct-score WITHIN each outcome
class so each class sums to its 1X2 prob. EV per scoreline using the
stage scoring table. Render an EV heatmap per match.

No model / no Poisson — everything is grounded in the listed bookie prices.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DATE = "2026-06-29"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATE)
os.makedirs(OUTDIR, exist_ok=True)

# Round-of-32 scoring
DIR_PTS = 2
EXACT_PTS = 5

# -------- matches: (home, away, 1X2 odds, correct-score list, kickoff IDT) -----
# correct-score entries: (home_goals, away_goals, decimal_odds)
MATCHES = [
    dict(
        home="Brazil", away="Japan", ko="Jun 29 20:00 IDT",
        odds=dict(home=1.70, draw=3.70, away=5.25),
        cs=[(1,0,6.50),(1,1,7.00),(2,0,7.50),(2,1,9.00),(0,0,8.50),
            (0,1,13.00),(3,0,13.00),(3,1,17.00),(1,2,19.00),(2,2,19.00),
            (0,2,29.00),(4,0,29.00),(3,2,34.00),(4,1,34.00),(1,3,41.00)],
    ),
    dict(
        home="Germany", away="Paraguay", ko="Jun 29 23:30 IDT",
        odds=dict(home=1.30, draw=5.25, away=11.00),
        cs=[(2,0,6.50),(1,0,7.50),(3,0,8.50),(2,1,9.50),(1,1,10.00),
            (3,1,12.00),(4,0,15.00),(0,0,13.00),(4,1,21.00),(0,1,21.00),
            (2,2,23.00),(1,2,26.00),(3,2,34.00),(5,0,29.00),(5,1,41.00)],
    ),
    dict(
        home="Netherlands", away="Morocco", ko="Jun 30 04:00 IDT",
        odds=dict(home=2.10, draw=3.00, away=2.60),
        cs=[(1,0,6.50),(2,0,11.00),(2,1,10.00),(3,0,21.00),(3,1,21.00),
            (3,2,34.00),(0,1,10.00),(0,2,17.00),(1,2,13.00),(0,3,41.00),
            (1,3,34.00),(2,3,41.00),(0,0,8.00),(1,1,6.00),(2,2,15.00)],
    ),
    dict(
        home="Cote d'Ivoire", away="Norway", ko="Jun 30 20:00 IDT",
        odds=dict(home=3.70, draw=3.50, away=2.00),
        cs=[(1,1,7.00),(0,1,9.00),(1,2,9.00),(0,2,11.00),(1,0,13.00),
            (2,1,13.00),(0,0,12.00),(2,2,13.00),(1,3,17.00),(0,3,19.00),
            (2,0,21.00),(2,3,26.00),(3,1,29.00),(3,2,34.00),(1,4,34.00)],
    ),
    dict(
        home="France", away="Sweden", ko="Jul 1 00:00 IDT",
        odds=dict(home=1.25, draw=6.00, away=12.00),
        cs=[(2,0,6.50),(3,0,8.50),(1,0,9.50),(2,1,10.00),(3,1,11.00),
            (1,1,12.00),(4,0,13.00),(4,1,17.00),(0,0,17.00),(2,2,23.00),
            (5,0,23.00),(3,2,26.00),(0,1,29.00),(1,2,29.00),(5,1,29.00)],
    ),
]


def devig_1x2(odds):
    raw = {k: 1.0 / v for k, v in odds.items()}
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}


def cls_of(hg, ag):
    if hg > ag:
        return "home"
    if hg < ag:
        return "away"
    return "draw"


def devig_cs(cs, p1x2):
    """De-vig correct-score probs within each outcome class to its 1X2 prob."""
    by_cls = {"home": [], "draw": [], "away": []}
    for hg, ag, o in cs:
        by_cls[cls_of(hg, ag)].append((hg, ag, 1.0 / o))
    probs = {}
    for c, items in by_cls.items():
        tot = sum(p for _, _, p in items)
        for hg, ag, p in items:
            probs[(hg, ag)] = (p / tot) * p1x2[c] if tot > 0 else 0.0
    return probs


def build(m):
    p1x2 = devig_1x2(m["odds"])
    # favorite by win prob (may be away)
    if p1x2["home"] >= p1x2["away"]:
        fav, dog, fav_cls = m["home"], m["away"], "home"
    else:
        fav, dog, fav_cls = m["away"], m["home"], "away"
    csp = devig_cs(m["cs"], p1x2)

    rows = []  # (fav_goals, dog_goals, label, p_exact, ev, cls)
    for (hg, ag), pe in csp.items():
        c = cls_of(hg, ag)
        p_class = p1x2[c]
        ev = DIR_PTS * p_class + (EXACT_PTS - DIR_PTS) * pe
        if fav_cls == "home":
            fg, dg = hg, ag
            label = f"{m['home']} {hg}-{ag}"
        else:
            fg, dg = ag, hg
            label = f"{m['home']} {hg}-{ag}"
        rows.append(dict(fg=fg, dg=dg, hg=hg, ag=ag, label=label,
                         pe=pe, ev=ev, cls=c))
    rows.sort(key=lambda r: r["ev"], reverse=True)
    return dict(m=m, p1x2=p1x2, fav=fav, dog=dog, fav_cls=fav_cls, rows=rows)


def heatmap(b):
    m = b["m"]
    fav, dog, fav_cls = b["fav"], b["dog"], b["fav_cls"]
    N = 6  # 0..5
    ev_grid = np.full((N, N), np.nan)
    pe_grid = np.full((N, N), np.nan)
    for r in b["rows"]:
        if 0 <= r["fg"] <= 5 and 0 <= r["dg"] <= 5:
            ev_grid[r["fg"], r["dg"]] = r["ev"]
            pe_grid[r["fg"], r["dg"]] = r["pe"]

    best = b["rows"][0]
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    masked = np.ma.masked_invalid(ev_grid)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#eeeeee")
    vmax = np.nanmax(ev_grid)
    im = ax.imshow(masked, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                   aspect="equal")

    for fg in range(N):
        for dg in range(N):
            if np.isnan(ev_grid[fg, dg]):
                ax.text(dg, fg, "–", ha="center", va="center",
                        color="#999", fontsize=9)
                continue
            ev = ev_grid[fg, dg]
            pe = pe_grid[fg, dg] * 100
            tcol = "white" if ev > vmax * 0.6 else "black"
            ax.text(dg, fg + 0.13, f"{ev:.2f}", ha="center", va="center",
                    color=tcol, fontsize=12, fontweight="bold")
            ax.text(dg, fg - 0.22, f"{pe:.0f}%", ha="center", va="center",
                    color=tcol, fontsize=8)

    # blue box around best-EV cell
    ax.add_patch(Rectangle((best["dg"] - 0.5, best["fg"] - 0.5), 1, 1,
                           fill=False, edgecolor="#1565c0", lw=3.5))
    # dotted draw diagonal (favorite goals == underdog goals)
    for d in range(N):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                               edgecolor="#222", lw=1.4, linestyle=":"))

    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xlabel(f"{dog} goals (underdog)", fontsize=11)
    ax.set_ylabel(f"{fav} goals (favorite)", fontsize=11)
    ax.set_title(f"{fav} v {dog} — best: {best['label']} (EV {best['ev']:.2f})",
                 fontsize=13, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("EV (points)")
    fig.tight_layout()
    slug = f"{m['home']}-{m['away']}".lower().replace(" ", "-").replace("'", "")
    path = os.path.join(OUTDIR, f"{slug}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    summary = []
    for m in MATCHES:
        b = build(m)
        path = heatmap(b)
        best = b["rows"][0]
        best_draw = next((r for r in b["rows"] if r["cls"] == "draw"), None)
        p = b["p1x2"]
        print("=" * 64)
        print(f"{m['home']} v {m['away']}  ({m['ko']})  R32")
        print(f"  de-vig 1X2:  {m['home']} {p['home']*100:.1f}% | "
              f"Draw {p['draw']*100:.1f}% | {m['away']} {p['away']*100:.1f}%")
        print(f"  favorite: {b['fav']}  (underdog {b['dog']})")
        print(f"  {'scoreline':<18}{'P(exact)':>10}{'EV':>8}")
        for r in b["rows"][:6]:
            print(f"  {r['label']:<18}{r['pe']*100:>9.1f}%{r['ev']:>8.2f}")
        print(f"  BEST PICK : {best['label']}  EV {best['ev']:.2f} "
              f"(P {best['pe']*100:.1f}%)")
        if best_draw:
            print(f"  BEST DRAW : {best_draw['label']}  EV {best_draw['ev']:.2f} "
                  f"(P {best_draw['pe']*100:.1f}%)")
        print(f"  heatmap -> {path}")
        summary.append((m, b, best, best_draw))

    print("\n" + "#" * 64)
    print("SUMMARY — best pick per match")
    print(f"{'match':<26}{'KO (IDT)':<18}{'best pick':<20}{'EV':>6}")
    for m, b, best, bd in summary:
        print(f"{m['home']+' v '+m['away']:<26}{m['ko']:<18}"
              f"{best['label']:<20}{best['ev']:>6.2f}")
    print("\nCouldn't price: Mexico v Ecuador (Jul 1 04:00 IDT) — "
          "kickoff.co.uk page not yet published (404).")
    print("Caveat: bet365 correct-score list is partial (~15 lines, no "
          "'any other score' tail); de-vigged within each class from the "
          "scorelines listed.")


if __name__ == "__main__":
    main()
