# Novelty check — adaptive PUEA detection

_Search run on 2026-10-05 against [OpenAlex](https://openalex.org).
This is a structured search, not a formal systematic review. Before
submitting, do a manual pass on IEEE Xplore and Google Scholar._

## Method

- **Keyword sweep:** 25 queries, 25 results each, giving 551 unique works.
  The queries covered:
  - PUEA with location, sequential tests, drift, poisoning and ML;
  - spectrum-sensing poisoning;
  - physical-layer authentication (PLA) under spoofing;
  - RSS location verification;
  - generic frog-boiling, drift-versus-poisoning and rollback.
- **Targeted sweep:** 12 more queries on the specific parts of the proposal.
- **Citation chasing:** all works citing 8 seed papers:
  - Chen et al. 2008 (665 citing works);
  - Jin et al. 2009;
  - Yan et al. 2017;
  - Chen et al. 2010;
  - Kravchik et al. 2021;
  - Chan-Tin et al. 2009;
  - Kloft & Laskov 2010;
  - Chhetry & Marchang 2021.
- **Screening:** citing works were filtered by keyword (poison, drift,
  sequential, CUSUM, change point, rollback and similar). Abstracts of close
  hits were read. Full texts were **not** read.

The scripts are in the session scratchpad and are not part of the repo. They
can be re-created from the queries listed above.

## What is already published

| Idea (ours) | Prior art | Status |
|---|---|---|
| RSS location check against the known PU position | Chen, Park & Reed 2008 [1] | Known |
| GLRT with transmit power fitted per slot (power-invariant check) | Yan et al. 2017 [3]: a GLRT with estimated power is equivalent to a differential LRT and asymptotically optimal | Known |
| Detection fails for an attacker near the PU | Ghanem et al. 2019 [20]: 24% detection at 50 m | Known. Our resolution sweep only adds numbers |
| Sequential tests for PUEA | Jin et al. 2009 [2] (WSPRT). Blesa et al. 2013 [6] (CUSUM). Sorrells et al. 2012 [21] (quickest detection) | Known |
| Sequential location verification | Zhang et al. 2019 [5] (VANET, double threshold) | Known |
| Spoofing detected as RSS coming from more than one location | Chen et al. 2010 [7] | Known |
| Gated self-update of a profile can be poisoned | Biggio et al. 2012/2013 [9, 10] (biometric templates). Kravchik et al. 2021 [8] (ICS detectors retrained for drift). Chan-Tin et al. 2009 [11]. Kloft & Laskov 2010 [12] | Known in general |
| Poisoning in spectrum sensing | Shi et al. 2018 [17] | Known |
| Adaptive updates gated on the authentication decision | Guo et al. 2026 [13] (channel-prediction PLA) | Known |
| Roll the model back when divergence is detected | Hallur et al. 2026 [14] (rollback of BEV latent memory) | Known in general |
| Drift and attacks modelled jointly | Hossain et al. 2026 [22] (learned latent model, IoT) | Known in general, learned rather than physics-based |
| Empirical threshold on how fast a drift can be before it is detected | Hong 2026 [15] (sharp threshold ε\*, fitted empirically) | Known empirically, no closed form |
| Correlated shadowing helps RSS location verification | Yan et al. 2016 [4] | Known. Relevant to our kill criterion |
| Adaptive / online PUEA detectors | Dong et al. 2018 [18]. Robert V et al. 2022 [19] | They exist. Their abstracts do not evaluate poisoning |

**Conclusion.** No single part of our current framework is novel: not the
physical gate, the adaptive profile, the sequential test or rollback. The
experimental findings in `adaptive_puea.md` §10 are mostly PUEA-specific
instances of known effects.

## What was not found

1. **A poisoning evaluation of adaptive PUEA detectors.** Adaptive PUEA
   detectors exist [18, 19], and poisoning of self-updating detectors is known
   in biometrics and ICS [8–10]. No paper found does the PUEA case: gated or
   naive adaptation, near-PU poisoners, legitimate drift.
2. **Using the physics to separate legitimate drift from poisoning.** Every
   drift-versus-poisoning paper found learns the distinction [8, 14, 22]. None
   uses the fact that, for a static PU under a known propagation model, all
   legitimate drift lies in a few nuisance parameters (P, N₀, σ) and location
   never changes.
3. **A closed-form feasibility boundary for poisoning.** Hong [15] fits the
   detection threshold empirically. No paper found derives detection delay
   and maximum contamination from the Fisher information of the sensing
   geometry.

## Proposed contribution: invariant-anchored adaptation

**Premise.** For a static PU, legitimate drift can change the transmit power
P, the noise floor N₀ and the shadowing spread σ. It cannot move the
source. An attacker displaced by δ ≠ 0 changes the spatial shape of the
RSS field across SUs, however carefully it matches power.

**Mechanism.**

1. *Adapt only the nuisance parameters.* Track P, N₀ and σ online, and freeze
   location. The adaptive profile then cannot move in any direction that
   physically corresponds to a different source position.
2. *Drift-invariant score.*
   - Per slot, compute the efficient score of the location at x_PU with P
     profiled out: `u_t = ∂/∂x log L(s_t | x, P̂_t)` at `x = x_PU`.
   - Under H₀, `E[u_t] = 0` and `Cov[u_t] = J`. J is the Fisher information,
     in closed form from the geometry, α̂ and σ̂.
   - Power drift has no effect on u_t. Shadowing drift only rescales J.
3. *Sequential test.*
   - Run a GLR-CUSUM on u_t, giving an alarm and an estimated change point k̂.
   - Calibrate on the PU-only stream for a target in-control run length, so
     no thresholds are set by hand.
4. *Rollback and attribution.*
   - On an alarm, restore the nuisance state to its checkpoint before k̂.
   - Estimate the attacker's location x̂_A from the slots since k̂.
   - Label each slot with a simple-vs-simple LRT between x_PU and x̂_A.
5. *Analytic boundary.*
   - For an attacker active in a fraction ρ of slots, displaced by δ: KL per
     slot ≈ ½ρ²·δᵀJδ (small δ).
   - Lorden's bound gives delay ≈ log(ARL₀) / KL.
   - With adaptation rate λ, contamination before the alarm is bounded by
     about λρ × delay, and it is confined to the nuisance parameters.
   - This gives a closed-form tradeoff between drift tracking and poisoning in
     terms of N_SU, σ, ρ, δ and λ, to be checked against the simulation.
     The derivation still has to be done carefully; the large-δ regime is
     nonlinear.

**Novelty claim (only if the experiments support it).** A physics-defined
split between parameters that may adapt and one that may not, combined with a
drift-invariant sequential test on the latter, plus a closed-form poisoning
boundary. Each building block has precedent ([3], [5], [6], [14]). The
combination and the boundary were not found.

### Outcome (2026-10-05)

The method was implemented and evaluated. Results, and the status of each
kill criterion, are in `adaptive_puea.md` §11. In summary:

- criterion 1 passed (the delay prediction is semi-analytic only);
- criterion 2 passed, with caveats;
- **criterion 3 failed**: the adaptive profile adds little over physics plus
  the sequential test;
- criterion 4 partly passed;
- criterion 5 was not tested.

The novelty claim is therefore narrowed to the **sequential drift-invariant
location test with rollback, and its geometry-based performance
prediction**. Invariant-anchored *adaptation* of a profile is not part of the
claim.

### Kill criteria (fixed before running)

| Test | If it fails |
|---|---|
| Analytic J predicts the measured single-slot resolution (`reports/adaptive/anchor_resolution.csv`) | Fix the theory before building anything |
| Gain survives `shadow_time_correlation` ≥ 0.9 | Averaging removes noise but not a static shadowing bias. Replace the path-loss anchor with a calibrated per-SU fingerprint (precedent: [4]), or report the result as limited to fast fading |
| Physical-Only + sequential test does not match the full system | If it matches, the adaptive profile contributes only drift handling, and the paper says so |
| Low-duty-cycle poisoner (ρ minimised) still caught before significant contamination | If not, the contribution is the tradeoff curve, not a defence |
| Attacker that inflates σ̂ to shrink J | σ̂ is the only adaptive input to the test. Bound its update rate and report the residual risk |

### Fundamental limits (state in any write-up)

- a co-located attacker (δ ≈ 0);
- multiple coordinated transmitters;
- a mobile PU, which breaks the premise that location never changes;
- simulation only. The propagation model matches the detector's own model.

## References (from OpenAlex)

1. R. Chen, J.-M. Park, J. H. Reed, "Defense against Primary User Emulation Attacks in Cognitive Radio Networks," IEEE JSAC, 2008. https://doi.org/10.1109/jsac.2008.080104
2. Z. Jin et al., "Detecting Primary User Emulation Attacks in Dynamic Spectrum Access Networks," ICC, 2009. https://doi.org/10.1109/icc.2009.5198911
3. S. Yan et al., "Location Verification Systems Based on Received Signal Strength With Unknown Transmit Power," IEEE Commun. Lett., 2017. https://doi.org/10.1109/lcomm.2017.2787129
4. S. Yan et al., "Location Verification Systems Under Spatially Correlated Shadowing," IEEE TWC, 2016. https://doi.org/10.1109/twc.2016.2535303
5. Y. Zhang et al., "Sequential Detection of Location Verification Based on Double Thresholds," IAEAC, 2019. https://doi.org/10.1109/iaeac47372.2019.8997593
6. J. Blesa et al., "PUE attack detection in CWSNs using anomaly detection techniques," EURASIP JWCN, 2013. https://doi.org/10.1186/1687-1499-2013-215
7. Y. Chen et al., "Detecting and Localizing Identity-Based Attacks in Wireless and Sensor Networks," IEEE TVT, 2010. https://doi.org/10.1109/tvt.2010.2044904
8. M. Kravchik et al., "Poisoning attacks on cyber attack detectors for industrial control systems," ACM SAC, 2021. https://doi.org/10.1145/3412841.3441892
9. B. Biggio et al., "Poisoning Adaptive Biometric Systems," S+SSPR, 2012. https://doi.org/10.1007/978-3-642-34166-3_46
10. B. Biggio et al., "Poisoning attacks to compromise face templates," ICB, 2013. https://doi.org/10.1109/icb.2013.6613006
11. E. Chan-Tin et al., "The Frog-Boiling Attack: Limitations of Anomaly Detection for Secure Network Coordinate Systems," SecureComm, 2009. https://doi.org/10.1007/978-3-642-05284-2_26
12. M. Kloft, P. Laskov, "Online Anomaly Detection under Adversarial Impact," AISTATS, 2010. https://openalex.org/W1513231349
13. Guo et al., "Channel Prediction-Based Physical Layer Authentication under Consecutive Spoofing Attacks," arXiv:2603.19962, 2026.
14. S. Hallur et al., "Slow Drift Temporal Poisoning attacks and vision language model guided defense for BEV perception in autonomous vehicles," 2026. https://doi.org/10.1007/s44465-026-00048-7
15. Z. Hong, "The Boiling Frog Threshold: Criticality and Blindness in World Model-Based Anomaly Detection Under Gradual Drift," arXiv:2603.08455, 2026.
16. A. W. Min et al., "Robust Tracking of Small-Scale Mobile Primary User in Cognitive Radio Networks," IEEE TPDS, 2012. https://doi.org/10.1109/tpds.2012.191
17. Y. Shi et al., "Spectrum Data Poisoning with Adversarial Deep Learning," MILCOM, 2018. https://doi.org/10.1109/milcom.2018.8599832
18. Q. Dong et al., "An Adaptive Primary User Emulation Attack Detection Mechanism for Cognitive Radio Networks," 2018. https://doi.org/10.1007/978-3-030-01701-9_17
19. N. J. Robert V et al., "OAM-GANN: Online Adaptive Memory based Genetically optimized ANN for PUEA Detection in CRN Applications," preprint, 2022. https://doi.org/10.21203/rs.3.rs-1952113/v2
20. W. R. Ghanem et al., "Particle Swarm Optimization Approaches for PUEA Detection and Localization in Cognitive Radio Networks," arXiv:1902.01944, 2019.
21. C. Sorrells et al., "Quickest detection of denial-of-service attacks in cognitive wireless networks," IEEE HST, 2012. https://doi.org/10.1109/ths.2012.6459913
22. M. K. Hossain et al., "Multi-Scale Sensor-Aware Variational Autoencoders for Adaptive IoT Security," IEEE Access, 2026. https://doi.org/10.1109/access.2026.3723588

---

## Round 2: deeper search (2026-10-06)

### What was searched

| Source | Queries / method |
|---|---|
| Web search (extended), restricted to ieeexplore.ieee.org | Two queries: PUEA with sequential detection, location, drift; and location verification with a sequential test, poisoning and an adaptive reference |
| Web search (extended), unrestricted | Nine queries covering sequential/CUSUM RSS location verification; PUEA with multi-slot evidence; closed-form LVS performance; poisoning of adaptive physical-layer authentication; PUEA drift and poisoning 2024–26; quickest-detection spoofing; sequential LVS near the claimed location; PUEA change-point/CUSUM; gradual spoofing against channel tracking |
| arXiv API (Boolean full-metadata search) | PUEA AND (sequential OR CUSUM OR poisoning OR drift); location verification AND (sequential OR CUSUM OR change detection); PLA AND (poisoning OR boiling frog OR rollback); spoofing detection AND RSS AND (CUSUM OR sequential OR quickest); spectrum sensing AND poisoning AND (adaptive OR online OR drift) |
| OpenAlex forward citations | Every work citing nine seed papers: Zhang 2019, Yan 2014/2016/2017, Blesa 2013, Xiao 2008 (388 citing works), Kravchik 2021, Ghanem 2019, Biggio 2012 |
| Full-text / abstract reading | Guo et al. 2026 (full HTML); arXiv 2101.06185, 1903.03684, 2310.11043, 2302.01841, 2502.16737, 1307.3348; WCSP 2018 LVS-RISE; EuroS&P 2020 biometric backdoors |
| Semantic Scholar API | Attempted. Rate-limited without an API key, so no usable results |

### New prior art found

| Work | What it does | Effect on our claim |
|---|---|---|
| Location Verification based on Radio Irregularity: Sequential Evaluation (WCSP 2018, doi 10.1109/wcsp.2018.8555939) | Sliding-window voting over RSS observations for location verification | A second precedent for multi-observation location verification. No drift handling, no adaptive reference, no poisoning |
| Yan & Malaney et al., *Optimal Information-Theoretic Wireless Location Verification* (2014) and the LVS papers [3, 4] | Closed-form detection and false-positive rates for single-shot RSS location verification | **Our single-slot resolution prediction is standard detection theory.** It is a sanity check, not a contribution |
| SPRT on RSS + location for malicious anchors in localization (several papers in the web results) | Sequential tests on RSS for secure localization | Sequential RSS testing is known in general |
| Lovisotto et al., *Biometric Backdoors* (EuroS&P 2020) | Incremental poisoning of unsupervised template updates, with a proposed defence | The poisoning-of-self-update threat and its defences are known in biometrics |
| Guo et al. 2026 [13], read in full | Updates gated on the authentication decision. The threat model is a fixed-location spoofer. **Gradual poisoning of the gated update, proximity attackers and state rollback are not analysed** | The closest wireless work. It leaves exactly our gap open |
| arXiv 2101.06185 (adaptive Kalman CSI spoofing detection), 2310.11043 (spoofing detection robust to user movement), 1903.03684 (Kalman PUEA with a mobile PU) | Adapt to channel or user dynamics | None analyses an attacker that exploits the adaptation, and none uses sequential location evidence |
| arXiv 2502.16737 (certified bounds for online dynamic poisoning) | Generic certified bounds for mean estimation and classification | Generic. Not wireless, no drift/poisoning separation |

### What was still not found

No paper found that combines, for PUEA or RSS location verification:

1. a **sequential** location test that is **invariant to legitimate drift** (power profiled out, shadowing variance tracked from a location-invariant residual); with
2. use of that test to **protect an adaptive detector from poisoning**, including **rollback** to the estimated change point; and
3. a **geometry-based prediction of the sequential detection delay**, validated against simulation.

The arXiv Boolean searches for these combinations returned 0–1 results each, and those results were off-topic.

### Revised assessment

- **Not novel:**
  - sequential location verification as such (2018, 2019 precedents);
  - SPRT/CUSUM on RSS;
  - closed-form single-shot location-verification performance (Yan et al.);
  - poisoning of self-updating systems (biometrics, ICS);
  - rollback on divergence (2026);
  - decision-gated wireless tracking (Guo 2026).
- **Plausibly novel:** components 1 + 2 + 3 together, in the PUEA / adaptive-detector setting, with the empirical demonstration that they close part of the near-PU poisoning gap. This is an integration-and-analysis contribution. It is not a new detection primitive.
- **Residual risk:**
  - IEEE Xplore and Google Scholar were not searched directly (no API access). The `site:ieeexplore.ieee.org` web searches only cover what the search engine indexes.
  - Paywalled full texts were not read.
  - **Before submission, the author should search IEEE Xplore and Google Scholar manually** for: "sequential location verification", "CUSUM" + "spoofing" + "RSS", "primary user emulation" + "sequential", and "physical layer authentication" + "poisoning". They should also check the citing works of Zhang 2019 and of the WCSP 2018 LVS-RISE paper.
