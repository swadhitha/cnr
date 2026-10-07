# Demo script (about 8 minutes)

Start the app from the repository root:

```bash
streamlit run puea_detection/app/demo_app.py
```

Pages are in the sidebar, in the order to present them.

| # | Page | What it shows | What to say |
|---|---|---|---|
| 1 | **Home** | The project in three cards | "An attacker pretends to be the licensed user. We detect it with machine learning, explain why, and add a new method for attackers standing right next to the real user." |
| 2 | **Check a signal** | One test signal, a big verdict (✅ Genuine / 🚨 Attack), confidence, whether it was right, and the top reasons | Click 🎲 a few times. "Orange bars push towards *attack*, blue towards *genuine*. Location mismatch usually decides it." |
| 3 | **Compare models** | Accuracy of the six models on 20,000 unseen signals | "XGBoost is best at 92.2%, with 3.7% false alarms." |
| 4 | **What the model uses** | The features the model relies on most | "Location mismatch is 37% of the model's attention. Attackers can copy power, but not position." |
| 5 | **Attack replay** | Old method vs new method on one simulated attack | Default: *poisoning from right next to the real user*, weak sensing. Point to the two result cards. In the chart, the black line rises past the dashed line, which raises the alarm. The ✖ on the map is the estimated attacker location. Then pick *No attack, the real user's power slowly changes* to show no extra false alarms. |
| 6 | **New method results** | Averages over 5 runs, a comparison with a rival method, what works and its limits | Read the three headline numbers: caught 12% → 63%, poisoned 29% → 8%, false alarms about 1% → 1%. Then read the ⚠️ Limits box out loud. |

## Likely questions

- **"What is new?"** Adding up location evidence over time, in a way that
  normal signal changes can't fool. It protects a learning detector from
  being poisoned, undoes the damage, and how fast it raises an alarm can be
  predicted from the sensor layout. The literature check is in
  `docs/novelty_lit_review.md`.
- **"Were the thresholds tuned on the test data?"** No. Every threshold comes
  from separate data with no attacks in it. The one change made after seeing
  results (keeping the alarm on) is labelled in `docs/adaptive_puea.md` §11.
- **"Is it always better?"** No. The *Freeze* version struggles if the
  attacker stays while the real user's power changes: pick *Stealthy
  attacker* to show this. The rival (Spatial-XGBoost) matches it on
  near-by attackers, but raises false alarms when signals change.
