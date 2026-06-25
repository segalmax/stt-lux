#!/usr/bin/env python3
"""WC2026 bookies-grounded EV picks + heatmaps.
Data: bet365 1X2 + correct-score from kickoff.co.uk (fetched manually).
De-vig 1X2 -> P(class). De-vig correct-score WITHIN each class to sum to its 1X2 prob.
Group stage scoring: direction=1, exact=3 -> EV = P(class) + 2*P(exact).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATE = "2026-06-25"

# Group-stage scoring
DIR_PTS, EXACT_PTS = 1, 3

# Each match: home, away, KO (IDT), 1X2 (home,draw,away decimals),
# correct scores as list of (cls, a, b, odds):
#   cls 'home': home scored a, away scored b
#   cls 'away': away scored a, home scored b
#   cls 'draw': a-b (a==b)
MATCHES = [
    dict(home="Curacao", away="Ivory Coast", ko="Jun25 23:00", x12=(16.00, 7.50, 1.14),
         cs=[('away',2,0,6.00),('away',3,0,6.50),('away',1,0,8.50),('away',4,0,9.50),
             ('away',2,1,11.00),('away',3,1,12.00),('draw',1,1,15.00),('away',5,0,17.00),
             ('away',4,1,17.00),('draw',0,0,17.00),('away',5,1,29.00),('away',6,0,29.00),
             ('home',1,0,34.00),('draw',2,2,34.00),('away',3,2,41.00)]),
    dict(home="Ecuador", away="Germany", ko="Jun25 23:00", x12=(2.80, 3.20, 1.25),
         cs=[('draw',1,1,8.00),('away',2,1,9.00),('away',1,0,9.00),('away',2,0,10.00),
             ('home',2,1,13.00),('away',3,1,15.00),('home',1,0,13.00),('draw',2,2,15.00),
             ('away',3,0,15.00),('draw',0,0,15.00),('away',3,2,23.00),('home',2,0,23.00),
             ('away',4,1,29.00),('away',4,0,29.00),('home',3,1,34.00)]),
    dict(home="Japan", away="Sweden", ko="Jun26 02:00", x12=(1.85, 3.50, 4.33),
         cs=[('draw',1,1,7.00),('home',2,1,9.00),('home',1,0,9.00),('home',2,0,10.00),
             ('away',2,1,15.00),('away',1,0,15.00),('home',3,1,15.00),('draw',2,2,13.00),
             ('draw',0,0,12.00),('home',3,0,17.00),('away',2,0,26.00),('home',3,2,23.00),
             ('home',4,1,29.00),('home',4,0,34.00),('away',3,1,34.00)]),
    dict(home="Tunisia", away="Netherlands", ko="Jun26 02:00", x12=(25.00, 8.50, 1.09),
         cs=[('away',3,0,6.00),('away',2,0,6.00),('away',4,0,8.00),('away',1,0,9.00),
             ('away',3,1,13.00),('away',2,1,13.00),('away',5,0,13.00),('away',4,1,17.00),
             ('draw',1,1,19.00),('away',6,0,23.00),('draw',0,0,19.00),('away',5,1,26.00),
             ('away',6,1,41.00),('away',3,2,41.00),('draw',2,2,41.00)]),
    dict(home="Turkey", away="USA", ko="Jun26 05:00", x12=(2.45, 3.10, 1.91),
         cs=[('draw',1,1,8.00),('away',2,1,9.00),('away',1,0,9.00),('away',2,0,11.00),
             ('home',2,1,13.00),('home',1,0,13.00),('draw',2,2,15.00),('away',3,1,15.00),
             ('draw',0,0,15.00),('away',3,0,17.00),('home',2,0,21.00),('away',3,2,23.00),
             ('home',3,1,29.00),('away',4,1,29.00),('home',3,2,34.00)]),
    dict(home="Paraguay", away="Australia", ko="Jun26 05:00", x12=(2.90, 2.25, 2.80),
         cs=[('home',1,0,5.50),('draw',0,0,4.00),('draw',1,1,5.00),('away',1,0,8.00),
             ('home',2,0,10.00),('home',2,1,11.00),('away',2,1,16.00),('away',2,0,18.00),
             ('home',3,0,25.00),('draw',2,2,16.00),('home',3,1,28.00),('away',3,1,40.00),
             ('away',3,0,40.00),('home',3,2,40.00),('home',4,0,50.00)]),
    dict(home="Norway", away="France", ko="Jun26 22:00", x12=(4.33, 4.50, 1.67),
         cs=[('draw',1,1,8.50),('away',2,1,8.50),('away',1,0,8.50),('away',2,0,9.00),
             ('away',3,1,13.00),('away',3,0,13.00),('home',2,1,15.00),('home',1,0,15.00),
             ('draw',2,2,17.00),('draw',0,0,15.00),('away',3,2,23.00),('away',4,1,26.00),
             ('home',2,0,26.00),('away',4,0,26.00),('home',3,1,34.00)]),
    dict(home="Senegal", away="Iraq", ko="Jun26 22:00", x12=(1.22, 5.50, 12.00),
         cs=[('home',2,0,7.00),('home',3,0,8.00),('home',1,0,8.50),('home',2,1,10.00),
             ('home',3,1,11.00),('draw',1,1,13.00),('home',4,0,12.00),('home',4,1,17.00),
             ('draw',0,0,17.00),('home',5,0,21.00),('draw',2,2,26.00),('away',1,0,29.00),
             ('home',3,2,29.00),('home',5,1,29.00),('away',2,1,34.00)]),
    dict(home="Cape Verde", away="Saudi Arabia", ko="Jun27 03:00", x12=(2.45, 3.50, 2.75),
         cs=[('draw',1,1,6.50),('home',1,0,8.50),('away',1,0,9.50),('home',2,1,10.00),
             ('away',2,1,11.00),('draw',0,0,10.00),('home',2,0,13.00),('away',2,0,15.00),
             ('draw',2,2,15.00),('home',3,1,21.00),('away',3,1,23.00),('home',3,0,26.00),
             ('away',3,0,29.00),('home',3,2,29.00),('away',3,2,34.00)]),
    dict(home="Uruguay", away="Spain", ko="Jun27 03:00", x12=(5.50, 3.20, 1.53),
         cs=[('away',1,0,7.00),('away',2,0,7.00),('draw',1,1,8.00),('away',2,1,9.00),
             ('away',3,0,11.00),('away',3,1,13.00),('draw',0,0,11.00),('home',1,0,19.00),
             ('home',2,1,21.00),('draw',2,2,19.00),('away',4,0,21.00),('away',4,1,26.00),
             ('away',3,2,29.00),('home',2,0,41.00),('away',5,0,41.00)]),
    dict(home="Egypt", away="Iran", ko="Jun27 06:00", x12=(2.45, 2.62, 3.75),
         cs=[('home',1,0,6.50),('draw',1,1,5.50),('draw',0,0,5.50),('away',1,0,8.50),
             ('home',2,0,11.00),('home',2,1,11.00),('away',2,1,15.00),('away',2,0,17.00),
             ('draw',2,2,19.00),('home',3,0,23.00),('home',3,1,26.00),('away',3,1,41.00),
             ('away',3,0,41.00),('home',3,2,41.00),('away',3,2,51.00)]),
    dict(home="New Zealand", away="Belgium", ko="Jun27 06:00", x12=(15.00, 8.50, 1.14),
         cs=[('away',2,0,6.50),('away',3,0,6.50),('away',1,0,8.50),('away',4,0,9.50),
             ('away',2,1,11.00),('away',3,1,12.00),('draw',1,1,17.00),('away',5,0,15.00),
             ('away',4,1,17.00),('draw',0,0,19.00),('away',5,1,26.00),('away',6,0,29.00),
             ('draw',2,2,34.00),('away',3,2,34.00),('home',1,0,41.00)]),
]


def to_hg_ag(cls, a, b):
    """Return (home_goals, away_goals) for a correct-score entry."""
    if cls == 'home':
        return a, b
    if cls == 'away':
        return b, a
    return a, b  # draw, a==b


def devig_1x2(home, draw, away):
    imp = np.array([1/home, 1/draw, 1/away])
    return imp / imp.sum()  # P(home), P(draw), P(away)


def process(m):
    ph, pd, pa = devig_1x2(*m['x12'])
    # Build scorelines with class membership
    scores = []  # (hg, ag, cls, implied)
    for cls, a, b, odds in m['cs']:
        hg, ag = to_hg_ag(cls, a, b)
        if hg > ag:
            klass = 'home'
        elif hg < ag:
            klass = 'away'
        else:
            klass = 'draw'
        scores.append([hg, ag, klass, 1/odds])
    class_prob = {'home': ph, 'draw': pd, 'away': pa}
    # De-vig within each class
    sums = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
    for hg, ag, klass, imp in scores:
        sums[klass] += imp
    p_exact = {}  # (hg,ag) -> P
    for hg, ag, klass, imp in scores:
        p = class_prob[klass] * imp / sums[klass] if sums[klass] > 0 else 0.0
        p_exact[(hg, ag)] = p
    # Favorite
    if pa > ph:
        fav, dog, fav_is = m['away'], m['home'], 'away'
        fav_prob, dog_prob = pa, ph
    else:
        fav, dog, fav_is = m['home'], m['away'], 'home'
        fav_prob, dog_prob = ph, pa
    m.update(dict(ph=ph, pd=pd, pa=pa, p_exact=p_exact, class_prob=class_prob,
                  fav=fav, dog=dog, fav_is=fav_is, fav_prob=fav_prob, dog_prob=dog_prob))
    return m


def cell_ev(m, fav_goals, dog_goals):
    """EV and P(exact) for a heatmap cell (favorite scores fav_goals, dog scores dog_goals)."""
    # Map (fav,dog) -> (home,away) goals
    if m['fav_is'] == 'home':
        hg, ag = fav_goals, dog_goals
    else:
        hg, ag = dog_goals, fav_goals
    if hg > ag:
        klass = 'home'
    elif hg < ag:
        klass = 'away'
    else:
        klass = 'draw'
    pcls = m['class_prob'][klass]
    pex = m['p_exact'].get((hg, ag), 0.0)
    ev = DIR_PTS * pcls + (EXACT_PTS - DIR_PTS) * pex  # = pcls + 2*pex
    return ev, pex


def build_grid(m):
    N = 6  # 0..5
    ev_grid = np.zeros((N, N))
    pex_grid = np.zeros((N, N))
    for f in range(N):       # favorite goals -> y
        for d in range(N):   # dog goals -> x
            ev, pex = cell_ev(m, f, d)
            ev_grid[f, d] = ev
            pex_grid[f, d] = pex
    return ev_grid, pex_grid


def ranked_scorelines(m):
    """Top scorelines by EV across listed scorelines (and all grid cells)."""
    rows = []
    for f in range(6):
        for d in range(6):
            ev, pex = cell_ev(m, f, d)
            rows.append((f, d, pex, ev))
    rows.sort(key=lambda r: -r[3])
    return rows


def plot_heatmap(m):
    ev_grid, pex_grid = build_grid(m)
    N = 6
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(ev_grid, origin='lower', cmap='YlOrRd', aspect='equal')
    # best cell
    bi = np.unravel_index(np.argmax(ev_grid), ev_grid.shape)
    best_f, best_d = bi  # f=y, d=x
    for f in range(N):
        for d in range(N):
            ev = ev_grid[f, d]
            pex = pex_grid[f, d] * 100
            ax.text(d, f + 0.16, f"{ev:.2f}", ha='center', va='center',
                    fontsize=11, fontweight='bold', color='black')
            ax.text(d, f - 0.22, f"{pex:.0f}%", ha='center', va='center',
                    fontsize=7.5, color='black')
            if f == d:  # draw diagonal dotted outline
                ax.add_patch(Rectangle((d-0.5, f-0.5), 1, 1, fill=False,
                                       edgecolor='gray', linestyle=':', linewidth=1.4))
    # blue box on best
    ax.add_patch(Rectangle((best_d-0.5, best_f-0.5), 1, 1, fill=False,
                           edgecolor='blue', linewidth=3))
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xlabel(f"{m['dog']} goals (underdog)", fontsize=11)
    ax.set_ylabel(f"{m['fav']} goals (favorite)", fontsize=11)
    best_ev = ev_grid[best_f, best_d]
    ax.set_title(f"{m['fav']} v {m['dog']} — best: {m['fav']} {best_f}-{best_d} (EV {best_ev:.2f})\n"
                 f"KO {m['ko']} IDT · P(fav)={m['fav_prob']*100:.0f}% draw={m['pd']*100:.0f}% "
                 f"dog={m['dog_prob']*100:.0f}%", fontsize=11)
    fig.colorbar(im, ax=ax, label="EV (points)", shrink=0.8)
    plt.tight_layout()
    slug = f"{m['fav']}_{m['dog']}".replace(' ', '-')
    path = os.path.join(OUTDIR, f"heatmap_{slug}.png")
    plt.savefig(path, dpi=130)
    plt.close()
    return path, (best_f, best_d, best_ev)


def main():
    summary = []
    for m in MATCHES:
        process(m)
        path, best = build_and_plot(m)
    print_report()


def build_and_plot(m):
    return plot_heatmap(m)


def fmt_score_fav(m, f, d):
    return f"{m['fav']} {f}-{d}" if f != d else f"Draw {f}-{d}"


def print_report():
    lines = []
    summary = []
    for m in MATCHES:
        path, (bf, bd, bev) = plot_heatmap(m)
        ranked = ranked_scorelines(m)
        # best pick
        best = ranked[0]
        # best draw
        draws = [r for r in ranked if r[0] == r[1]]
        best_draw = draws[0]
        lines.append(f"\n## {m['fav']} v {m['dog']}  (KO {m['ko']} IDT)")
        lines.append(f"P(win {m['fav']})={m['fav_prob']*100:.0f}%  P(draw)={m['pd']*100:.0f}%  "
                     f"P(win {m['dog']})={m['dog_prob']*100:.0f}%")
        lines.append("Top 6 by EV (fav-dog scoreline | P(exact) | EV):")
        for r in ranked[:6]:
            f, d, pex, ev = r
            label = f"{m['fav']} {f}-{d}" if f > d else (f"{m['dog']} {d}-{f}" if d > f else f"Draw {f}-{d}")
            lines.append(f"  {label:28s} P={pex*100:4.1f}%  EV={ev:.3f}")
        bf_, bd_, bpex, bev_ = best
        blabel = f"{m['fav']} {bf_}-{bd_}" if bf_ > bd_ else (f"{m['dog']} {bd_}-{bf_}" if bd_ > bf_ else f"Draw {bf_}-{bd_}")
        df_, dd_, dpex, dev_ = best_draw
        lines.append(f"BEST PICK: {blabel}  (EV {bev_:.3f}, P {bpex*100:.1f}%)")
        lines.append(f"BEST DRAW: Draw {df_}-{dd_}  (EV {dev_:.3f}, P {dpex*100:.1f}%)")
        summary.append((m, blabel, bev_, f"Draw {df_}-{dd_}", dev_))
    lines.append("\n\n# SUMMARY — best pick per match")
    lines.append(f"{'Match':28s} {'KO IDT':12s} {'Best pick':22s} {'EV':>6s}  {'Best draw':10s} {'EVd':>5s}")
    for m, blabel, bev, dlabel, dev in summary:
        match = f"{m['fav']} v {m['dog']}"
        lines.append(f"{match:28s} {m['ko']:12s} {blabel:22s} {bev:6.3f}  {dlabel:10s} {dev:5.3f}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUTDIR, "report.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    for m in MATCHES:
        process(m)
    print_report()
