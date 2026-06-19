#!/usr/bin/env python3
"""World Cup 2026 bookies-grounded EV picks + heatmaps. bet365 odds via kickoff.co.uk."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json, os

OUT = os.path.dirname(os.path.abspath(__file__))

# Scoring: group stage
DIR_PTS, EXACT_PTS = 1, 3  # both matches are group stage

def devig_1x2(odds):
    """odds = {'home':,'draw':,'away':} decimal -> de-vigged probs."""
    imp = {k: 1.0/v for k, v in odds.items()}
    s = sum(imp.values())
    return {k: v/s for k, v in imp.items()}, s

def devig_cs(cs_by_class, p_class):
    """cs_by_class: {'home':{(f,d):odds}, 'draw':{...}, 'away':{...}}
    Normalize implied probs WITHIN each class so they sum to p_class[class]."""
    out = {}
    for cls, lines in cs_by_class.items():
        imp = {sc: 1.0/o for sc, o in lines.items()}
        s = sum(imp.values())
        for sc, v in imp.items():
            out[sc] = (v/s) * p_class[cls]   # sc keyed as (fav_goals, dog_goals)
    return out

def cell_class(f, d):
    return 'home' if f > d else ('draw' if f == d else 'away')

def ev(f, d, p_class, p_exact):
    cls = cell_class(f, d)
    pc = p_class[cls]
    pe = p_exact.get((f, d), 0.0)
    return DIR_PTS * pc + (EXACT_PTS - DIR_PTS) * pe

def build_match(name, fav, dog, odds_1x2, cs_lines):
    """cs_lines: list of (fav_goals, dog_goals, decimal_odds). class inferred."""
    p_class, overround = devig_1x2(odds_1x2)
    # p_class keyed home/draw/away where home=favorite-win (favorite is home in both)
    cs_by_class = {'home': {}, 'draw': {}, 'away': {}}
    for f, d, o in cs_lines:
        cs_by_class[cell_class(f, d)][(f, d)] = o
    p_exact = devig_cs(cs_by_class, p_class)

    # EV grid 0..5 x 0..5
    G = 6
    ev_grid = np.zeros((G, G))
    pe_grid = np.zeros((G, G))
    for f in range(G):
        for d in range(G):
            ev_grid[f, d] = ev(f, d, p_class, p_exact)
            pe_grid[f, d] = p_exact.get((f, d), 0.0)

    best = np.unravel_index(np.argmax(ev_grid), ev_grid.shape)  # (f,d)

    # ranked scorelines by EV among priced lines
    ranked = sorted(
        [((f, d), p_exact[(f, d)], ev(f, d, p_class, p_exact)) for (f, d) in p_exact],
        key=lambda x: -x[2])

    # best draw
    draws = sorted([((f, d), pe, e) for (f, d), pe, e in ranked if f == d], key=lambda x: -x[2])

    return dict(name=name, fav=fav, dog=dog, p_class=p_class, overround=overround,
                p_exact=p_exact, ev_grid=ev_grid, pe_grid=pe_grid, best=best,
                ranked=ranked, best_draw=draws[0] if draws else None)

def heatmap(m, fname):
    fav, dog = m['fav'], m['dog']
    ev_grid, pe_grid, best = m['ev_grid'], m['pe_grid'], m['best']
    G = ev_grid.shape[0]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(ev_grid, cmap='YlOrRd', origin='lower', aspect='equal')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('EV (points)')

    for f in range(G):
        for d in range(G):
            e = ev_grid[f, d]
            pe = pe_grid[f, d]
            # contrast text color
            tc = 'black' if e < ev_grid.max() * 0.62 else 'white'
            ax.text(d, f + 0.13, f"{e:.2f}", ha='center', va='center',
                    fontsize=12, fontweight='bold', color=tc)
            ax.text(d, f - 0.22, f"{pe*100:.1f}%", ha='center', va='center',
                    fontsize=8, color=tc)

    # blue box on best
    bf, bd = best
    ax.add_patch(Rectangle((bd - 0.5, bf - 0.5), 1, 1, fill=False,
                           edgecolor='blue', lw=3.5))
    # dotted diagonal (draws)
    for i in range(G):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor='dimgray', lw=1.6, linestyle=':'))

    ax.set_xticks(range(G)); ax.set_yticks(range(G))
    ax.set_xlabel(f"{dog} (underdog) goals", fontsize=11)
    ax.set_ylabel(f"{fav} (favorite) goals", fontsize=11)
    ax.set_title(f"{fav} v {dog} — best: {fav} {bf}-{bd} (EV {ev_grid[best]:.2f})",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

# ---------------- DATA (bet365 via kickoff.co.uk, fetched 2026-06-20) ----------------
# scoreline tuples are (favorite_goals, underdog_goals); favorite = home team here
M1 = build_match(
    "Netherlands v Sweden (Group F, 20:00 IDT)", "Netherlands", "Sweden",
    {'home': 1.70, 'draw': 4.10, 'away': 4.50},
    [(2,1,9.00),(1,0,8.50),(2,0,9.00),(3,1,13.00),(3,0,15.00),(3,2,23.00),
     (4,1,26.00),(4,0,29.00),                          # home (NED) wins
     (1,1,8.00),(2,2,15.00),(0,0,13.00),               # draws
     (1,2,15.00),(0,1,15.00),(0,2,26.00),(1,3,41.00)], # away (SWE) wins
)
M2 = build_match(
    "Germany v Cote d'Ivoire (Group E, 23:00 IDT)", "Germany", "Cote d'Ivoire",
    {'home': 1.53, 'draw': 4.50, 'away': 5.50},
    [(2,0,9.00),(2,1,8.50),(1,0,9.00),(3,0,12.00),(3,1,12.00),(4,0,21.00),
     (4,1,21.00),(3,2,23.00),(4,2,41.00),              # home (GER) wins
     (1,1,9.00),(2,2,17.00),(0,0,15.00),               # draws
     (1,2,19.00),(0,1,19.00),(0,2,34.00)],             # away (CIV) wins
)

matches = [M1, M2]
files = ["heatmap_netherlands_sweden.png", "heatmap_germany_coteivoire.png"]
summary = []
for m, fn in zip(matches, files):
    p = heatmap(m, fn)
    bf, bd = m['best']
    summary.append((m['name'], f"{m['fav']} {bf}-{bd}", m['ev_grid'][m['best']], p))
    print("="*70)
    print(m['name'])
    print(f"  de-vig 1X2 (overround {(m['overround']-1)*100:.1f}%): "
          f"{m['fav']} {m['p_class']['home']*100:.1f}% | "
          f"Draw {m['p_class']['draw']*100:.1f}% | "
          f"{m['dog']} {m['p_class']['away']*100:.1f}%")
    print(f"  Top 6 scorelines by EV:")
    print(f"    {'scoreline':<22}{'P(exact)':>10}{'EV':>8}")
    for (f, d), pe, e in m['ranked'][:6]:
        cls = cell_class(f, d)
        label = (f"{m['fav']} {f}-{d}" if cls=='home'
                 else (f"Draw {f}-{d}" if cls=='draw' else f"{m['dog']} {d}-{f}"))
        print(f"    {label:<22}{pe*100:>9.1f}%{e:>8.3f}")
    (bf2, bd2) = m['best']
    print(f"  BEST PICK: {m['fav']} {bf2}-{bd2}  (EV {m['ev_grid'][m['best']]:.3f}, "
          f"P(exact) {m['pe_grid'][m['best']]*100:.1f}%)")
    if m['best_draw']:
        (df, dd), dpe, de = m['best_draw']
        print(f"  BEST DRAW: {df}-{dd}  (EV {de:.3f}, P(exact) {dpe*100:.1f}%)")
    print(f"  heatmap -> {p}")

print("\n" + "="*70)
print("SUMMARY — best pick + EV per match")
for name, pick, e, _ in summary:
    print(f"  {name:<48} {pick:<18} EV {e:.3f}")
