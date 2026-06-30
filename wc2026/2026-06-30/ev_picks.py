#!/usr/bin/env python3
"""WC2026 bookies-grounded EV picks + heatmaps (bet365 via kickoff.co.uk).

Round of 32 scoring: direction 2 pts, exact 5 pts.
EV(scoreline) = dir_pts * P(outcome class) + (exact_pts - dir_pts) * P(exact).
De-vig: 1X2 implied (1/odds) normalized to 1; correct-score implied normalized
WITHIN each outcome class so the class sums to its 1X2 prob. The fetched
correct-score list omits a small "any other score" tail -> de-vig only from the
scorelines listed (caveat).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DATE = "2026-06-30"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DIR_PTS, EXACT_PTS = 2, 5  # Round of 32

# Each match: home/away names, 1X2 decimal odds, correct-score {(home_goals, away_goals): odds}
MATCHES = [
    {
        "home": "Cote d'Ivoire", "away": "Norway", "ko": "Jun 30, 20:00 IDT",
        "odds_1x2": {"home": 3.50, "draw": 3.50, "away": 2.10},
        "cs": {(1,1):6.50,(0,1):8.50,(1,2):9.50,(0,2):11.00,(1,0):12.00,(2,1):13.00,
               (0,0):11.00,(2,2):15.00,(1,3):17.00,(2,0):19.00,(0,3):19.00,(2,3):29.00,
               (3,1):29.00,(3,2):34.00,(1,4):41.00},
    },
    {
        "home": "France", "away": "Sweden", "ko": "Jul 1, 00:00 IDT",
        "odds_1x2": {"home": 1.29, "draw": 5.75, "away": 11.00},
        "cs": {(2,0):7.50,(3,0):9.00,(2,1):9.50,(1,0):10.00,(3,1):11.00,(1,1):11.00,
               (4,0):13.00,(4,1):17.00,(2,2):21.00,(5,0):23.00,(3,2):26.00,(0,0):19.00,
               (1,2):29.00,(5,1):29.00,(0,1):29.00},
    },
    {
        "home": "Mexico", "away": "Ecuador", "ko": "Jul 1, 04:00 IDT",
        "odds_1x2": {"home": 2.15, "draw": 2.90, "away": 4.00},
        "cs": {(1,0):5.50,(0,0):5.50,(1,1):6.00,(0,1):8.50,(2,0):9.00,(2,1):11.00,
               (1,2):17.00,(0,2):19.00,(3,0):21.00,(2,2):23.00,(3,1):26.00,(1,3):41.00,
               (3,2):41.00,(0,3):51.00,(4,0):51.00},
    },
    {
        "home": "England", "away": "Congo DR", "ko": "Jul 1, 19:00 IDT",
        "odds_1x2": {"home": 1.29, "draw": 5.50, "away": 11.00},
        "cs": {(2,0):5.50,(1,0):6.00,(3,0):7.50,(2,1):11.00,(1,1):11.00,(0,0):10.00,
               (4,0):13.00,(3,1):15.00,(0,1):21.00,(4,1):23.00,(5,0):29.00,(2,2):34.00,
               (1,2):34.00,(3,2):41.00,(5,1):41.00},
    },
    {
        "home": "Belgium", "away": "Senegal", "ko": "Jul 1, 23:00 IDT",
        "odds_1x2": {"home": 2.20, "draw": 3.20, "away": 3.60},
        "cs": {(1,1):6.00,(1,0):7.50,(0,1):10.00,(2,1):10.00,(0,0):8.00,(2,0):11.00,
               (1,2):13.00,(0,2):19.00,(2,2):15.00,(3,1):21.00,(3,0):21.00,(1,3):34.00,
               (3,2):34.00,(0,3):41.00,(2,3):41.00},
    },
    {
        "home": "USA", "away": "Bosnia", "ko": "Jul 2, 03:00 IDT",
        "odds_1x2": {"home": 1.40, "draw": 5.00, "away": 8.00},
        "cs": {(2,0):6.50,(1,0):7.00,(2,1):9.50,(3,0):9.00,(1,1):9.50,(3,1):13.00,
               (0,0):12.00,(4,0):17.00,(0,1):19.00,(4,1):23.00,(2,2):23.00,(1,2):23.00,
               (3,2):34.00,(5,0):34.00,(0,2):41.00},
    },
]


def devig_1x2(o):
    imp = {k: 1.0 / v for k, v in o.items()}
    s = sum(imp.values())
    return {k: v / s for k, v in imp.items()}


def cls(h, a):
    return "home" if h > a else ("away" if a > h else "draw")


def devig_cs(cs, p1x2):
    """Normalize correct-score implied probs within each class to its 1X2 prob."""
    imp = {sc: 1.0 / od for sc, od in cs.items()}
    cls_sum = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for (h, a), p in imp.items():
        cls_sum[cls(h, a)] += p
    out = {}
    for (h, a), p in imp.items():
        c = cls(h, a)
        out[(h, a)] = p / cls_sum[c] * p1x2[c] if cls_sum[c] > 0 else 0.0
    return out


def analyze(m):
    p1x2 = devig_1x2(m["odds_1x2"])
    pcs = devig_cs(m["cs"], p1x2)
    # favorite by win prob
    fav_is_home = p1x2["home"] >= p1x2["away"]
    fav = m["home"] if fav_is_home else m["away"]
    dog = m["away"] if fav_is_home else m["home"]
    p_fav = p1x2["home"] if fav_is_home else p1x2["away"]
    p_dog = p1x2["away"] if fav_is_home else p1x2["home"]
    p_draw = p1x2["draw"]

    # build grid in (fav_goals, dog_goals) orientation
    rows = []  # (fav_g, dog_g, p_exact, ev)
    for fg in range(6):
        for dg in range(6):
            # convert to home/away scoreline
            h, a = (fg, dg) if fav_is_home else (dg, fg)
            c = cls(h, a)
            pclass = {"home": p1x2["home"], "draw": p1x2["draw"], "away": p1x2["away"]}[c]
            pex = pcs.get((h, a), 0.0)
            ev = DIR_PTS * pclass + (EXACT_PTS - DIR_PTS) * pex
            rows.append((fg, dg, pex, ev))
    return {
        "m": m, "fav": fav, "dog": dog, "fav_is_home": fav_is_home,
        "p_fav": p_fav, "p_dog": p_dog, "p_draw": p_draw, "p1x2": p1x2,
        "pcs": pcs, "rows": rows,
    }


def heatmap(res):
    m = res["m"]
    fav, dog = res["fav"], res["dog"]
    ev = np.zeros((6, 6))
    pex = np.zeros((6, 6))
    for fg, dg, pe, e in res["rows"]:
        ev[fg, dg] = e
        pex[fg, dg] = pe
    best = max(res["rows"], key=lambda r: r[3])
    bfg, bdg = best[0], best[1]

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(ev, origin="lower", cmap="YlOrRd", aspect="auto")
    for fg in range(6):
        for dg in range(6):
            ax.text(dg, fg + 0.16, f"{ev[fg,dg]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black")
            ax.text(dg, fg - 0.24, f"{pex[fg,dg]*100:.0f}%", ha="center", va="center",
                    fontsize=7.5, color="#333333")
    # draw diagonal dotted outline
    for d in range(6):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                               edgecolor="gray", linestyle=":", linewidth=1.4))
    # blue box around best EV cell
    ax.add_patch(Rectangle((bdg - 0.5, bfg - 0.5), 1, 1, fill=False,
                           edgecolor="blue", linewidth=3))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel(f"{dog} goals (underdog)", fontsize=11)
    ax.set_ylabel(f"{fav} goals (favorite)", fontsize=11)
    ax.set_title(f"{fav} v {dog} — best: {fav} {bfg}-{bdg} (EV {best[3]:.2f})\n"
                 f"KO {m['ko']}  ·  bet365 de-vigged  ·  R32",
                 fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("EV (points)")
    fig.tight_layout()
    fn = os.path.join(OUTDIR, f"{m['home']}_v_{m['away']}".replace(" ", "_")
                      .replace("'", "") + ".png")
    fig.savefig(fn, dpi=130)
    plt.close(fig)
    return fn, best


def fmt_score(res, fg, dg):
    """Render fav-goals/dog-goals as a real scoreline string fav X-Y dog."""
    return f"{res['fav']} {fg}-{dg} {res['dog']}"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    summary = []
    for m in MATCHES:
        res = analyze(m)
        fn, best = heatmap(res)
        rows = sorted(res["rows"], key=lambda r: r[3], reverse=True)
        print("\n" + "=" * 64)
        print(f"{res['fav']} (fav) v {res['dog']}   KO {m['ko']}")
        print(f"  de-vig 1X2: {m['home']} {res['p1x2']['home']*100:.1f}% | "
              f"Draw {res['p1x2']['draw']*100:.1f}% | "
              f"{m['away']} {res['p1x2']['away']*100:.1f}%")
        print(f"  Top 6 scorelines by EV:")
        print(f"  {'scoreline':<28}{'P(exact)':>10}{'EV':>8}")
        for fg, dg, pe, e in rows[:6]:
            print(f"  {fmt_score(res,fg,dg):<28}{pe*100:>8.1f}%{e:>8.2f}")
        # best draw
        draws = [r for r in rows if r[0] == r[1]]
        bd = draws[0]
        print(f"  BEST PICK : {fmt_score(res,best[0],best[1])}  EV {best[3]:.2f}  "
              f"(P exact {best[2]*100:.1f}%)")
        print(f"  BEST DRAW : {fmt_score(res,bd[0],bd[1])}  EV {bd[3]:.2f}  "
              f"(P exact {bd[2]*100:.1f}%)")
        print(f"  saved: {os.path.basename(fn)}")
        summary.append((res, best, bd, os.path.basename(fn)))

    print("\n" + "#" * 64)
    print("SUMMARY — best pick + EV per match")
    print(f"{'match':<26}{'KO (IDT)':<18}{'best pick':<20}{'EV':>6}")
    for res, best, bd, fn in summary:
        match = f"{res['fav']} v {res['dog']}"
        pick = f"{best[0]}-{best[1]}"
        print(f"{match:<26}{res['m']['ko']:<18}{res['fav']+' '+pick:<20}{best[3]:>6.2f}")
    print("\nCaveat: correct-score de-vigged only from the ~15 bet365 lines listed "
          "per match (no 'any other score' tail).")


if __name__ == "__main__":
    main()
