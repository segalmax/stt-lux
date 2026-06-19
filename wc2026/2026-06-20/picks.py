#!/usr/bin/env python3
"""WC2026 bookies-grounded EV picks + heatmaps for 2026-06-20 (IDT).

Data: bet365 odds via kickoff.co.uk. No model/Poisson.
Scoring (group stage): direction 1pt, exact 3pt -> EV = P(class) + 2*P(exact).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Group stage scoring
DIR_PTS, EXACT_PTS = 1, 3


def devig_1x2(home, draw, away):
    imp = np.array([1/home, 1/draw, 1/away])
    return imp / imp.sum()  # P(home), P(draw), P(away)


def devig_class(lines, class_prob):
    """lines: dict scoreline->odds. Normalize implied within class to class_prob."""
    imp = {k: 1/v for k, v in lines.items()}
    s = sum(imp.values())
    return {k: (v/s)*class_prob for k, v in imp.items()}


def ev_for(p_exact, class_prob):
    return DIR_PTS*class_prob + (EXACT_PTS-DIR_PTS)*p_exact


def build(match):
    ph, pd, pa = devig_1x2(*match["x12"])
    p1x2 = {"home": ph, "draw": pd, "away": pa}
    # favorite by win prob
    fav_is_home = ph >= pa
    fav_name = match["home"] if fav_is_home else match["away"]
    dog_name = match["away"] if fav_is_home else match["home"]
    fav_class_prob = ph if fav_is_home else pa

    # de-vig each class
    p_home = devig_class(match["home_scores"], ph)
    p_draw = devig_class(match["draw_scores"], pd)
    p_away = devig_class(match["away_scores"], pa)

    # Build unified scoreline prob in (fav_goals, dog_goals) orientation
    probs = {}   # (fav, dog) -> p_exact
    classp = {}  # (fav, dog) -> class prob
    def add(fg, dg, p, cp):
        probs[(fg, dg)] = probs.get((fg, dg), 0) + p
        classp[(fg, dg)] = cp

    for (h, a), p in [(parse(k), v) for k, v in p_home.items()]:
        fg, dg = (h, a) if fav_is_home else (a, h)
        add(fg, dg, p, ph)
    for k, p in p_away.items():
        # away-score key "X-Y" = away team scored X, home scored Y
        x, y = parse(k)
        h, a = y, x  # convert to home-away orientation
        fg, dg = (h, a) if fav_is_home else (a, h)
        add(fg, dg, p, pa)
    for (h, a), p in [(parse(k), v) for k, v in p_draw.items()]:
        add(h, a, p, pd)  # draw symmetric

    return dict(match=match, p1x2=p1x2, fav_is_home=fav_is_home,
                fav_name=fav_name, dog_name=dog_name, fav_class_prob=fav_class_prob,
                ph=ph, pd=pd, pa=pa, probs=probs, classp=classp)


def parse(score):
    # "2-1" -> (2,1) home-away
    h, a = score.split("-")
    return int(h), int(a)


def heatmap(b, fname):
    fav, dog = b["fav_name"], b["dog_name"]
    N = 6
    EV = np.full((N, N), np.nan)
    PE = np.zeros((N, N))
    for (fg, dg), p in b["probs"].items():
        if fg < N and dg < N:
            cp = b["classp"][(fg, dg)]
            EV[fg, dg] = ev_for(p, cp)
            PE[fg, dg] = p
    # fill missing EV with class-only minimal (no exact contribution) where class known?
    # leave NaN cells as 0 EV for display
    EVdisp = np.nan_to_num(EV, nan=0.0)

    # best cell
    bi = np.unravel_index(np.argmax(EVdisp), EVdisp.shape)
    best_fg, best_dg = bi

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(EVdisp, origin="lower", cmap="YlOrRd", aspect="equal")
    for fg in range(N):
        for dg in range(N):
            ev = EVdisp[fg, dg]
            ax.text(dg, fg+0.16, f"{ev:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="black")
            ax.text(dg, fg-0.22, f"{PE[fg,dg]*100:.1f}%", ha="center", va="center",
                    fontsize=7, color="#333333")
    # blue box best
    ax.add_patch(Rectangle((best_dg-0.5, best_fg-0.5), 1, 1, fill=False,
                           edgecolor="blue", lw=3))
    # dotted draw diagonal
    for d in range(N):
        ax.add_patch(Rectangle((d-0.5, d-0.5), 1, 1, fill=False,
                               edgecolor="black", lw=1.2, linestyle=":"))
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xlabel(f"{dog} goals (underdog)")
    ax.set_ylabel(f"{fav} goals (favorite)")
    ax.set_title(f"{fav} v {dog} — best: {fav} {best_fg}-{best_dg} (EV {EVdisp[bi]:.3f})")
    fig.colorbar(im, ax=ax, label="EV")
    fig.tight_layout()
    path = os.path.join(OUTDIR, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, (best_fg, best_dg), EVdisp[bi]


def ranked(b, n=6):
    rows = []
    for (fg, dg), p in b["probs"].items():
        cp = b["classp"][(fg, dg)]
        rows.append((fg, dg, p, ev_for(p, cp)))
    rows.sort(key=lambda r: -r[3])
    return rows[:n]


MATCHES = [
    dict(
        home="Netherlands", away="Sweden", ko="20:00 IDT",
        x12=(1.73, 3.10, 3.50),
        home_scores={"2-1":8.0,"1-0":7.5,"2-0":8.0,"3-1":12.0,"3-0":14.0,
                     "3-2":22.0,"4-1":25.0,"4-0":28.0},
        draw_scores={"1-1":7.0,"2-2":14.0,"0-0":12.0},
        away_scores={"2-1":14.0,"1-0":14.0,"2-0":25.0,"3-1":40.0},
    ),
    dict(
        home="Germany", away="Cote d'Ivoire", ko="23:00 IDT",
        x12=(1.53, 3.50, 4.50),
        home_scores={"2-0":9.0,"2-1":8.5,"1-0":9.0,"3-0":12.0,"3-1":12.0,
                     "4-0":21.0,"4-1":21.0,"3-2":23.0,"4-2":41.0},
        draw_scores={"1-1":9.0,"2-2":17.0,"0-0":15.0},
        away_scores={"2-1":19.0,"1-0":19.0,"2-0":34.0},
    ),
]

summary = []
for i, m in enumerate(MATCHES, 1):
    b = build(m)
    fname = m["home"].lower().replace(" ", "-").replace("'", "") + "_vs_" + \
            m["away"].lower().replace(" ", "-").replace("'", "") + ".png"
    path, best, bev = heatmap(b, fname)
    rows = ranked(b)
    # best draw
    draws = [(fg, dg, p, ev_for(p, b["classp"][(fg,dg)]))
             for (fg, dg), p in b["probs"].items() if fg == dg]
    draws.sort(key=lambda r: -r[3])
    bd = draws[0]

    print(f"\n=== Match {i}: {b['fav_name']} v {b['dog_name']} (KO {m['ko']}) ===")
    print(f"De-vig 1X2: {m['home']} {b['ph']*100:.1f}% | Draw {b['pd']*100:.1f}% | {m['away']} {b['pa']*100:.1f}%")
    print(f"Favorite: {b['fav_name']} ({b['fav_class_prob']*100:.1f}% win)")
    print("Top 6 by EV  (scoreline = fav-dog):")
    print(f"{'score':>8} {'P(exact)':>9} {'EV':>7}")
    for fg, dg, p, ev in rows:
        print(f"{fg}-{dg:>6} {p*100:>7.1f}% {ev:>7.3f}")
    print(f"BEST PICK: {b['fav_name']} {best[0]}-{best[1]} (EV {bev:.3f})")
    print(f"BEST DRAW: {bd[0]}-{bd[1]} (P {bd[2]*100:.1f}%, EV {bd[3]:.3f})")
    print(f"Heatmap -> {path}")
    summary.append((b['fav_name'], b['dog_name'], best, bev, m['ko'],
                    bd, b['fav_name']))

print("\n\n===== SUMMARY ACROSS MATCHES =====")
print(f"{'Match':<28} {'KO':<10} {'Best pick':<18} {'EV':>6} {'Best draw':>12}")
for fav, dog, best, bev, ko, bd, favn in summary:
    bp = f"{favn} {best[0]}-{best[1]}"
    print(f"{fav+' v '+dog:<28} {ko:<10} {bp:<18} {bev:>6.3f} {str(bd[0])+'-'+str(bd[1]):>12}")
