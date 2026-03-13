# Hypothesis Hierarchy: Local Dendritic Signals as Proxies for Synaptic Gradients

---

## Motivation

Synapses must solve the credit assignment problem during learning: knowing how much to change their conductance $g_{syn}$ and in what direction. In gradient-based learning, this requires knowing $\partial L / \partial g_{syn}$ — the gradient of a loss with respect to synaptic conductance. But a synapse only has access to local signals: voltage $V(t)$ and calcium traces that carry information about three physiological events — the EPSP at presynaptic input, NMDA dendritic spiking, and the backpropagating action potential (bAP).

By factoring the full gradient:

$$
\frac{\partial L}{\partial g_{syn}} = \frac{\partial L}{\partial V^S} \cdot \frac{\partial V^S}{\partial g_{syn}}
$$

the second term — the Jacobian $J(t) = \partial V^S / \partial g_{syn}$ — captures the neuron's causal structure independently of any task or loss. If a synapse can approximate $J(t)$ using only local signals, it is doing most of the work of credit assignment through its own biophysics, without needing explicit knowledge of a global error signal.

---

## Foundational Mathematics

Synaptic current is parameterized as:

$$
I_{syn}(t) = g_{syn} \cdot s(t) \cdot (V(t) - E_{syn})
$$

where $s(t)$ is the synaptic activation variable governed by channel kinetics, and $V(t)$ is the local dendritic voltage at the synapse, which contains both EPSP and bAP signals. The Jacobian is a vector of dimension $1 \times T$:

$$
J(t) = \frac{\partial V^S(t)}{\partial g_{syn}}
$$

$$
J(t) = \frac{\partial V^S(t)}{\partial I_{syn}(t)} \frac{\partial I_{syn}(t)}{\partial g_{syn}}
$$


**Critical distinction:** $V(t)$ in the decoder input is local dendritic voltage, while $V^S(t)$ in the Jacobian is somatic voltage. These are not the same signal. The Jacobian encodes how a conductance perturbation at this synapse propagates through the full dendritic tree — through nonlinear channel dynamics and interactions with other inputs — to affect somatic voltage. Predicting $J(t)$ from local $V(t)$ is therefore a nontrivial mapping, not an algebraic identity.

Because $g_{syn}$ enters the dynamical system only through $I_{syn}(t)$, the Jacobian accumulates signal proportionally to:

$$
J(t) = \frac{\partial V^S(t)}{\partial I_{syn}(t)} \cdot s(t) \cdot (V(t) - E_{syn})
$$

$$
J(t) \propto s(t) \cdot (V(t) - E_{syn})
$$

---

## Derived Consequences

The following consequences follow analytically from the parameterization above. They are not assumptions and do not require empirical verification, though they should be confirmed as sanity checks in Jaxley before running any decoder analysis.

### Consequence 1 — Synaptic activation gates the Jacobian

$$
F = 0 \implies s(t) = 0 \; \forall t \implies J(t) = 0 \; \forall t
$$

A synapse that was not activated in a given trial has zero Jacobian throughout that trial, regardless of any other event including bAP. This is purely mathematical.

### Consequence 2 — bAP interacts with $J(t)$ only during the $s(t) \neq 0$ window

$$
J(t) \neq 0 \text{ only when } s(t) \neq 0
$$

The bAP modulates $V(t)$, but $V(t)$ only enters $J(t)$ during the window where $s(t) \neq 0$. The EPSP × bAP interaction is therefore bounded by the kinetics of $s(t)$, making the hypothesis timescale-dependent:

- **AMPA:** $s(t)$ decays in ~2–5 ms — the bAP may arrive after $s(t) \approx 0$ at distal synapses, particularly given nonlinear propagation delay through intervening dendritic morphology and conductances
- **NMDA:** $s(t)$ decays in ~50–100 ms — the window is wide enough for bAP to interact with $s(t)$ even at distal synapses

### Consequence 3 — N1a is analytically ruled out

bAP enters $J(t)$ only through $V(t) \cdot s(t)$. There is no pathway by which bAP contributes to $J(t)$ independently of $s(t)$. Therefore bAP alone cannot produce a nonzero Jacobian without a co-occurring EPSP. This is not a null hypothesis to test empirically — it is precluded by model construction.

---

## Hypothesis Hierarchy

Hypotheses are tested using a **pre-specified sequential strategy**: downstream hypotheses are evaluated only conditional on the outcome of upstream tests. This structure is established before data collection to prevent post-hoc analysis.

### Primary Hypothesis (H1)

> The conjunction of feedforward local dendritic voltage dynamics (EPSP) and feedback via the backpropagating action potential (bAP) carries sufficient information to allow a local decoder to approximate the Jacobian $J(t) = \partial V^S(t) / \partial g_{syn}$ from the local voltage trace $V(t)$ at the synapse.

**Notes to consider**: 
- EPSP is part of V(t), and the context of other synapses may impact V(t) s.t. V(t) - E_syn --> 0
- s(t), which is the conductance of the synapse, is also involved in this conjunction proposed in H1. Must include this in further considerations.

**Biological motivation:** The EPSP establishes $s(t) \neq 0$; the bAP modulates $V(t)$ during that window. Their interaction is therefore the primary carrier of $J(t)$ information available locally. This interaction is gated: only when a synapse is active and a bAP arrives does the EPSP × bAP product contribute to $J(t)$.

**Operationally:** A 1D CNN decoder trained on $V(t) \in \mathbb{R}^T$ to predict $J(t) \in \mathbb{R}^T$ achieves $R^2$ exceeding the 97.5th percentile of both null distributions (Null 1 and Null 2), with a 95% CI that excludes the median of each null distribution, evaluated on held-out stimulus patterns via cross-validation.

### Secondary Hypothesis (H2)

> Including NMDA conductance improves decodability of $J(t)$ compared to AMPA alone.

**Motivation:** From Consequence 2, the EPSP × bAP interaction requires $s(t) \neq 0$ during bAP arrival. NMDA kinetics extend $s(t)$ long enough for this overlap to occur at distal synapses given nonlinear bAP propagation delays. AMPA kinetics decay too rapidly to guarantee this overlap.

**Operationally:** $R^2$ under AMPA+NMDA exceeds $R^2$ under AMPA alone on the same held-out patterns, and both exceed their respective null distributions at the 97.5th percentile. Tested only if H1 is supported.

### Null Hypotheses

#### N1a — bAP alone is sufficient to encode $J(t)$

*Analytically ruled out by Consequence 3.* bAP can only enter $J(t)$ through $s(t) \cdot V(t)$. When $s(t) = 0$, bAP has no mathematical pathway to influence $J(t)$. This is excluded by model construction, not empirical test.

*Sanity check:* Confirm that $J(t) \approx 0$ for all inactive synapses ($F = 0$) across all trials including bAP trials, verifying Consequence 1 holds numerically in the Jaxley implementation.

**Note to consider**:
- If contextual synaptic activity increases V(t) when s(t) = 0, that is not much different from EPSP occuring. If that is the case, then how would we isolate this condition s.t. bAP alone is occuring?
- If for a given synapse bAP alone is occuring and contextual V(t) depolarization is present, then bAP is not necessarily alone?
- If V(t) alone in general can be used to encode J(t), then s(t), especially when s(t) = 0 when EPSP should not be occuring, doesn't matter for encoding Jacobian

#### N1b — EPSP alone is sufficient to encode $J(t)$

> If EPSP alone, without bAP arrival at the synapse, is sufficient to decode $J(t)$, then bAP contributes no additional information and H1 is not supported.

N1b is operationalized using **trials where bAP does not reach the synapse of interest**, giving $V(t)$ with EPSP features but no bAP signal. Two mechanistically distinct conditions provide this:

| Condition | Mechanism | What $V(t)$ contains | Expected $J(t)$ |
| --- | --- | --- | --- |
| $F=1$, no AP fired | Subthreshold excitation | EPSP only | Small, EPSP-driven |
| $F=1$, bAP shunted | On-path inhibitory shunting | EPSP only, conductance-altered $V(t)$ | Small, possibly different from above |

These two conditions are not equivalent. Shunting inhibition directly alters local membrane conductance and $V(t)$, while a subthreshold trial leaves $V(t)$ unperturbed by inhibition. If the Jacobian differs between these two N1b conditions, that difference is itself informative about the mechanism of inhibitory gating.

**Note on operationalization:** N1b cannot be tested by holding out bAP features from $V(t)$ in trials where a bAP occurred, because EPSP and bAP signals co-occur in the same continuous trace and cannot be cleanly separated without arbitrary assumptions. Trial selection — not feature ablation — is the correct operationalization.

#### N1c — Neither EPSP nor bAP encodes $J(t)$

> Local signals at the synapse are insufficient to approximate $J(t)$. The decoder performs at chance across all conditions.

This is the **primary null tested first** in the sequential strategy. All subsequent tests are conditional on its rejection.

#### N2 — AMPA alone is sufficient; NMDA is unnecessary

> Decodability under AMPA alone is not significantly different from AMPA+NMDA.

Tested only if H1 is supported.

---

## Experimental Design

### Stimulus Structure

- One trial = one simulation of duration $T$ ms, with one synchronous input pattern (a subset of synapses co-activated at variable locations along the dendritic tree)
- Patterns are **balanced**: half evoke a somatic AP (and bAP), half do not
- Inhibitory perturbations are applied on-path to create bAP-shunted conditions within AP trials, providing the mechanistic N1b condition naturally within the dataset
- Each synapse must appear across many trials in both $F=1$ and $F=0$ conditions — within-synapse variation in activation status is required to prevent the decoder from learning a fixed anatomical location → Jacobian mapping rather than a trial-specific $V(t) \rightarrow J(t)$ mapping
- Pattern diversity must be sufficient to support cross-validation by held-out patterns; the decoder must generalize to novel input contexts

### Data Point Structure

One data point = one synapse on one trial:

- **Input X:** local voltage trace $V(t) \in \mathbb{R}^T$ at the synapse, containing EPSP and bAP events
- **Target Y:** Jacobian $J(t) \in \mathbb{R}^T$, computed via autodiff in Jaxley as $\partial V^S(t) / \partial g_{syn}$

At $\Delta t = 0.25$ ms and $T = 100$ ms, both $V(t)$ and $J(t)$ are vectors of length 400. The decoder is a sequence-to-sequence regression: $f: \mathbb{R}^{400} \rightarrow \mathbb{R}^{400}$.

### Cross-Validation

Entire stimulus patterns are held out for the test set — individual data points are never split across train and test sets. This tests whether the learned $V(t) \rightarrow J(t)$ mapping generalizes to novel input contexts, rather than memorizing pattern-specific statistics.

---

## Decoder Architecture

A **1D Convolutional Neural Network (1D CNN)** is used as the primary decoder:

$$
\hat{J}(t) = \mathbf{w}_{out} \cdot \text{ReLU}\left(W_{conv} * V(t)\right) + b
$$

**One convolutional layer:** $K$ filters of length $L$ applied to $V(t)$. Each filter detects a local temporal pattern — EPSP onset shape, bAP arrival shape, or their temporal overlap. Filter length $L$ must span at least the full EPSP-bAP interaction window; for NMDA conditions this requires $L$ covering ~50–100 ms.

**One pointwise nonlinearity (ReLU):** Justified by two known nonlinearities in the system:

1. bAP propagation amplitude is modulated nonlinearly by intervening synaptic conductances, which vary across patterns
2. $J(t) \propto s(t) \cdot V(t)$ is a product of two time-varying signals — a nonlinearity a linear filter cannot represent

ReLU is natural because $J(t) \geq 0$ when $V(t) > E_{syn}$ and $s(t) > 0$.

**One linear readout layer:** Maps $K$ filter activations at each timepoint to a scalar $\hat{J}(t)$.

### Why 1D CNN Over Alternatives

**Temporal translation equivariance is a structural requirement for interpretable null testing.** The 1D CNN learns what the EPSP-bAP interaction pattern looks like independently of when in the trial it occurs. This property is required for Null 2 (circular shift) to be interpretable: a decoder without translation equivariance (e.g. a flat MLP) would fail the circular shift null because it learned the mapping at specific timepoints and cannot generalize across shifts — not because the shift broke the causal signal. Using a flat MLP would make Null 2 uninterpretable as a test of the $s(t)$ confound.

**Linear regression with time-lagged features** is a strict special case ($K=1$ filter, no nonlinearity) used as a **lower bound**. If even a linear model achieves above-null decodability, that is informative. Significant improvement of the 1D CNN over linear regression provides evidence that nonlinear bAP propagation dynamics are necessary to capture the $V(t) \rightarrow J(t)$ mapping.

**Interpretability:** Each learned filter $w^k_\tau$ is a temporal kernel that can be visualized directly. If learned filters align with EPSP and bAP timescales, this constitutes mechanistic evidence for the interaction structure predicted by H1.

---

## Null Dataset Construction

Two null datasets are constructed, each targeting a distinct confound. Both must be exceeded for H1 to be supported.

### Null 1: Stratified Permutation Null

**Confound addressed:** Spurious cross-trial distributional associations. A decoder could achieve high $R^2$ by learning marginal statistics — for example, “whenever bAP=ON the Jacobian is large for every synapse on this trial” — without learning the trial-specific EPSP×bAP→$J(t)$ mapping. Pattern memorization is addressed by holding out entire patterns during cross-validation; the permutation null addresses residual distributional associations within the training set.

**What it breaks:** The trial-specific pairing between $V(t)$ and $J(t)$ across trials, within each stratum.

**What it preserves:** Marginal distributions of $V(t)$ and $J(t)$ for each synapse; gating structure ($F=1$ vs $F=0$); known bAP→$J$ and AP→$J$ distributional relationships.

**Stratification:** Permutation is performed within strata defined by trial-level binary events:

| Stratum | $F$ | bAP at synapse |
| --- | --- | --- |
| 1 | 1 | 1 |
| 2 | 1 | 0 |
| 3 | 0 | 1 |
| 4 | 0 | 0 |

Permuting across strata would mix Jacobians from fundamentally different trial types, making the null trivially easy to beat. Permuting within strata preserves all known distributional organization while destroying only the within-condition, trial-specific $V(t) \rightarrow J(t)$ coupling that H1 claims exists.

**Procedure** — for each synapse:

1. Assign each trial to its stratum based on $F$ and bAP status
2. Within each stratum, randomly shuffle $J(t)$ labels across trials while keeping $V(t)$ in place
3. Combine shuffled strata into the null dataset
4. Train the decoder and record $R^2$ on held-out patterns
5. Repeat 1000 times to build a null distribution
6. Compare real decoder $R^2$ to the 97.5th percentile of this distribution

### Null 2: Circular Shift Null

**Confound addressed:** Spurious within-trial temporal co-activation. $s(t)$ is a common cause of both $V(t)$ and $J(t)$ within a single trial, creating a backdoor path:

$$
V(t) \leftarrow s(t) \rightarrow J(t)
$$

Specifically: $s(t)$ drives the EPSP in $V(t)$, making $V(t)$ nonzero when $s(t) \neq 0$; and $s(t)$ gates $J(t) \propto s(t) \cdot (V(t) - E_{syn})$, making $J(t)$ nonzero only when $s(t) \neq 0$. Both signals are therefore active during the same temporal window — not because local voltage causally encodes the Jacobian via dendritic propagation (H1's claim), but simply because $s(t)$ activates both simultaneously. A decoder could exploit this shared temporal support without learning anything about the EPSP×bAP interaction.

**What the circular shift does:** Within each trial, $J(t)$ is circularly shifted by a fixed offset $\Delta t$:

$$
J_{shifted}(t) = J\left((t + \Delta t) \bmod T\right)
$$

This displaces $J(t)$'s nonzero window away from the $s(t) \neq 0$ window that $V(t)$ aligns with, breaking the shared temporal support while preserving:

- The autocorrelation structure of $J(t)$
- The autocorrelation structure of $V(t)$
- The marginal distributions of both signals

**Shift magnitude $\Delta t$:** $\Delta t$ must be large enough to fully displace $J(t)$'s nonzero window outside the $s(t) \neq 0$ window. This is synapse-type dependent and must be verified empirically:

- **AMPA condition:** $\Delta t \gtrsim 5$–$10$ ms (larger than AMPA $s(t)$ decay timescale)
- **NMDA condition:** $\Delta t \gtrsim 50$–$100$ ms (larger than NMDA $s(t)$ decay timescale)

These are **separate circular shift nulls** for the AMPA-only and AMPA+NMDA conditions and are not interchangeable.

**Why translation equivariance is required for this null to be interpretable:** A decoder without temporal translation equivariance (e.g. a flat MLP) fails the circular shift null because it learned the mapping at specific timepoints and cannot generalize across temporal shifts — not because the shift broke the causal signal. The 1D CNN's translation equivariance ensures that performance on the circular shift null reflects the causal structure of the problem, not a limitation of the decoder.

**Interpretation:**

- Performance **collapses** after shift → decoder was exploiting shared $s(t)$ temporal support. H1 is not supported.
- Performance **survives** shift → decoder learned structure that generalizes beyond instantaneous co-activation, consistent with the true causal path $V(t) \rightarrow J(t)$ via dendritic propagation.

**Procedure** — for each synapse:

1. Record real $V(t)$ and $J(t)$ for each trial
2. Circularly shift $J(t)$ by $\Delta t$ to produce $J_{shifted}(t)$
3. Train decoder on $V(t) \rightarrow J_{shifted}(t)$, using the same cross-validation structure as the real decoder
4. Record $R^2$ on held-out patterns
5. Repeat across a range of $\Delta t$ values to confirm the null is stable
6. Compare real decoder $R^2$ to the 97.5th percentile of the shift null distribution

### Summary: Two Nulls, Two Distinct Confounds

| Null | Confound | What it breaks | What it preserves |
| --- | --- | --- | --- |
| Null 1: Stratified permutation | Cross-trial distributional associations | Trial-specific $V(t) \rightarrow J(t)$ pairing within strata | Marginal distributions, bAP→$J$ and AP→$J$ relationships |
| Null 2: Circular shift | Within-trial shared $s(t)$ temporal support | Instantaneous $V(t) \leftrightarrow J(t)$ temporal alignment | Autocorrelation structure of both signals |

Both nulls must be exceeded for H1 to be supported.

---

## Full Experimental Condition Table

| Condition | $F$ | bAP at synapse | Mechanism | Expected $J(t)$ | Tests |
| --- | --- | --- | --- | --- | --- |
| Inactive synapse | 0 | irrelevant | $s(t)=0$ by Consequence 1 | $\approx 0$ | Sanity check for N1a |
| Active, no AP trial | 1 | 0 | Subthreshold excitation | Small, EPSP-driven | N1b |
| Active, bAP shunted | 1 | 0 | On-path inhibitory gating | Small, conductance-altered $V(t)$ | Mechanistic N1b |
| Active, bAP present | 1 | 1 | Full EPSP × bAP interaction | Large | H1 |

---

## Decoder Comparison Table

| Decoder | Input $V(t)$ | Target | Trials | Tests |
| --- | --- | --- | --- | --- |
| 1D CNN | Full $V(t)$, AMPA+NMDA | Real $J(t)$ | All $F=1$ | H1 primary |
| 1D CNN | Full $V(t)$, AMPA+NMDA | Real $J(t)$ | $F=1$, bAP=0 only | N1b |
| 1D CNN | Full $V(t)$, AMPA+NMDA | Stratified permuted $J(t)$ | All $F=1$ | Null 1 |
| 1D CNN | Full $V(t)$, AMPA+NMDA | Circularly shifted $J(t)$, $\Delta t_{NMDA}$ | All $F=1$ | Null 2 (NMDA) |
| 1D CNN | Full $V(t)$, AMPA only | Real $J(t)$ | All $F=1$ | N2 / H2 comparison |
| 1D CNN | Full $V(t)$, AMPA only | Circularly shifted $J(t)$, $\Delta t_{AMPA}$ | All $F=1$ | Null 2 (AMPA) |
| Linear regression | Full $V(t)$, AMPA+NMDA | Real $J(t)$ | All $F=1$ | Lower bound |

---

## Sequential Testing Strategy and Decision Criteria

All success criteria use $R^2$ evaluated on held-out patterns. Success is defined as exceeding the 97.5th percentile of the relevant null distribution with a 95% CI that excludes the null median.

### Step 1 — Test N1c

Run the primary 1D CNN decoder on all $F=1$ trials with AMPA+NMDA. Compare $R^2$ against both null distributions.

> If $R^2_{real}$ fails to exceed the 97.5th percentile of either null distribution: **N1c is not rejected.** Local signals are insufficient to decode $J(t)$. Stop.

> If $R^2_{real}$ exceeds the 97.5th percentile of both null distributions with 95% CI excluding each null median: proceed to Step 2.

### Step 2 — Test N1b

Run the decoder on N1b trials only ($F=1$, bAP=0, combining subthreshold and shunted conditions). Compare $R^2_{N1b}$ to $R^2_{full}$.

> If $R^2_{N1b} \approx R^2_{full}$: EPSP alone is sufficient. **N1b is not rejected.** H1 is not supported. Stop.

> If $R^2_{N1b} \ll R^2_{full}$: bAP contributes information beyond EPSP alone. Proceed to Step 3.

### Step 3 — Accept H1

H1 is supported if all three of the following hold simultaneously on held-out patterns:

$$
R^2_{\text{full}} \gg R^2_{\text{N1b decoder}}
$$

$$
R^2_{\text{full}} > \text{97.5th percentile of } R^2_{\text{Null 1 (permutation)}}
$$

$$
R^2_{\text{full}} > \text{97.5th percentile of } R^2_{\text{Null 2 (circular shift)}}
$$

The first criterion establishes that bAP contributes information beyond EPSP alone. The second establishes that the decoder learned a generalizable cross-trial mapping rather than exploiting marginal distributional statistics. The third establishes that the mapping reflects genuine dendritic propagation structure rather than spurious within-trial temporal co-activation shared between $V(t)$ and $J(t)$ through $s(t)$.

### Step 4 — Test N2 / H2 (conditional on H1)

Compare $R^2$ under AMPA+NMDA versus AMPA alone, each against their respective circular shift nulls.

> If $R^2_{AMPA+NMDA} > R^2_{AMPA}$ and both exceed the 97.5th percentile of their respective Null 2 distributions: **reject N2, accept H2.**

> If $R^2_{AMPA+NMDA} \approx R^2_{AMPA}$: **N2 is not rejected.** NMDA kinetics are unnecessary for decodability at the timescales studied.

---

## Sanity Checks

Before running any decoder analysis, confirm the following empirically in Jaxley:

1. **Consequence 1:** $J(t) \approx 0$ for all inactive synapses ($F=0$) across all trials, including bAP trials. Numerical near-zero is expected from floating point; verify the magnitude is negligible relative to active synapse Jacobians.
2. **Consequence 2:** $J(t)$ is nonzero only during the $s(t) \neq 0$ window. Verifies that the temporal support of $J(t)$ is correctly bounded by channel kinetics.
3. **Shift validity for Null 2:** For each synapse type (AMPA, NMDA), verify that the chosen $\Delta t$ fully displaces $J_{shifted}(t)$'s nonzero window outside the $s(t) \neq 0$ window before running the circular shift null.
