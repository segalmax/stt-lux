# World Cup 2026 — Daily EV Picks Routine (prompt)

Paste the block below into the routine's prompt field (Claude Code web → Routines →
edit → Prompt). Schedule: every morning, timezone Asia/Jerusalem.

---

TASK: Daily World Cup 2026 bookies-grounded EV picks + heatmaps.

TIME / SCOPE
- Determine "today" with `TZ=Asia/Jerusalem date "+%Y-%m-%d %H:%M %Z"`. Do NOT trust the
  container's system date or any injected currentDate — they can be a day off; the shell
  command above is the source of truth.
- Find all FIFA World Cup 2026 matches whose kickoff falls on TODAY's Jerusalem calendar
  date AND is still in the future vs the current Jerusalem time (upcoming only).
- Schedules are usually published in ET/BST. Convert every kickoff to IDT before deciding:
  IDT = ET + 7 = BST + 2. Beware late "midnight ET" games — those roll into the NEXT
  Jerusalem day and must be EXCLUDED from today.
- If there are no upcoming matches today, send NOTHING (silence = no matches) — just reply
  "No upcoming World Cup matches today" in the session and stop. Do not push a notification.

DATA (bookies only — never use a model/Poisson)
- For each match, fetch CURRENT bet365 odds from kickoff.co.uk:
  /world-cup-predictions-stats-odds/<home>-vs-<away>/  — the 1X2 and the full
  correct-score market.
- Slug quirks: use full English country names; Ivory Coast = `cote-d-ivoire`. If a guessed
  slug 404s, web-search `site:kickoff.co.uk <home> <away> world-cup-predictions-stats-odds`
  to find the exact URL, then fetch that.
- De-vig the 1X2 -> P(home win), P(draw), P(away win) (implied = 1/odds, normalize to sum 1).
- De-vig the correct-score prices -> P(each scoreline), by normalizing implied probs WITHIN
  each outcome class (home-win / draw / away-win) so each class sums to its 1X2 prob.
- The fetched correct-score list may omit a small "any other score" tail; de-vig from the
  scorelines actually listed and note this caveat. If a match can't be priced, list it as
  "couldn't price" and continue.

FAVORITE
- favorite = the team with the higher de-vigged win prob. It may be the AWAY team — if so,
  the favorite-win class is P(away win) and the heatmap's favorite axis is the away team.
  Orient EV/heatmap by win-prob, not by home/away.

SCORING (group / R32 / R16 / QF / SF / 3rd / final):
  direction pts: 1 / 2 / 2 / 4 / 5 / 5 / 8
  exact pts:     3 / 5 / 5 / 8 /10 /10 /15
  For a predicted scoreline:
    EV = direction_pts * P(outcome class) + (exact_pts - direction_pts) * P(exact)
  (group stage = P(class) + 2*P(exact)). Pick the right stage per match.

OUTPUT, per match (render with code execution, save images under wc2026/<YYYY-MM-DD>/):
1) HEATMAP: grid favorite goals (y, 0..5) vs underdog goals (x, 0..5); color by EV
   (yellow->red); each cell big number = EV, small number = P(exact) %; blue box around the
   highest-EV cell; dotted outline on the draw diagonal; title
   "<Fav> v <Dog> — best: <Fav> X-Y (EV Z)".
2) Ranked table: top 6 scorelines by EV (scoreline, P(exact), EV).
3) Single best pick + best DRAW scoreline for contrast.
Then ONE summary table across all matches: best pick + EV per match. Keep prose minimal,
decision-ready.

DELIVERY (this is the point of the run — nobody is watching the session)
- The notification IS the deliverable. Push the heatmap images with a PROACTIVE file
  delivery (SendUserFile, status: proactive). There is no separate PushNotification tool in
  this env — the proactive file delivery is what reaches the phone + inbox.
- Lead the caption/banner with the actual decisions, e.g.:
  "WC2026 picks — <date>. ① <Fav> X-Y vs <Dog> (EV Z, KO HH:MM IDT). ② ... Best draws: ..."
  so the picks are readable in the banner without opening the session.
- Commit the script + images to wc2026/<date>/ on the working branch and push.

---

## Learnings captured (2026-06-20 first run)
- System/injected date was 2026-06-19 while Asia/Jerusalem was already 2026-06-20 02:12 IDT
  → always derive the date from `TZ=Asia/Jerusalem date`.
- kickoff.co.uk slug for Ivory Coast is `germany-vs-cote-d-ivoire` (not `ivory-coast`).
- bet365 correct-score list from the page is partial (~15 lines, no "any other" bucket);
  de-vig within class from what's listed and note the caveat.
- ET→IDT = +7, BST→IDT = +2. The 8pm/midnight ET slots land on the next Jerusalem day.
- Proactive SendUserFile is the delivery channel; put picks in the caption, not just charts.
