//! Channel coupling: the redundancy discount that keeps the fusion from
//! double-counting channels that share a nuisance factor.
//!
//! Pairwise redundancy is read on the full-scored anchor, over the items both channels
//! actually scored, then assembled into weights through a regularized dimensionless
//! covariance. Two diagnostics are read from set membership only, so they cannot
//! inherit the live pool's collider bias.

// Index-based loops are the clearest form for the small matrix arithmetic here, matching
// the `linalg` module's convention; a single index addresses several rows at once.
#![allow(clippy::needless_range_loop)]

use crate::config::CouplingConfig;
use crate::ingest::anchor::Anchor;
use crate::keys::UnorderedPair;
use crate::summary::MeanVar;
use crate::weighting::NEUTRAL_WEIGHT;
use crate::weighting::linalg::{inverse_spd, solve_spd};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

/// One channel pair's redundancy correlation measured on the anchor, with the overlap
/// backing it.
///
/// Marked `#[non_exhaustive]`: a result type produced by [`anchor_correlations`] that
/// callers read but never construct.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PairObservation {
    /// The anchor redundancy correlation for this pair, over the both-scored items:
    /// Spearman's rank correlation mapped to the Gaussian-copula linear correlation
    /// by [`anchor_correlations`], in `[-1, 1]`.
    pub correlation: f64,
    /// The number of items both channels scored (the overlap backing the estimate).
    pub n_both: usize,
}

/// One channel pair's accumulated redundancy baseline, as consumed by
/// [`coupled_weights`]: the pooled correlation summary plus how many anchor refreshes
/// back it.
///
/// This is the weighting-stage projection of the persistent
/// [`PairSummary`](crate::state::PairSummary); the [`Fuser`](crate::Fuser) builds it, so
/// the estimator stays independent of the state layer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairBaseline {
    /// The accumulated redundancy correlation: overlap-weighted mean (the point
    /// estimate), total overlap count (the reliability), and overlap-weighted variance
    /// across refreshes and strata (the stability).
    pub redundancy: MeanVar,
    /// How many anchor refreshes contributed. Stability is a between-refresh property:
    /// one refresh has zero between-refresh variance by construction, so the discount is
    /// gated on [`CouplingConfig::min_refreshes`].
    pub refreshes: f64,
}

/// Per-channel weights after the redundancy discount, plus the effective
/// independent-channel count.
///
/// Marked `#[non_exhaustive]`: a result type produced by [`coupled_weights`] that
/// callers read but never construct.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CoupledWeights {
    /// The per-channel weights after the redundancy discount, normalized to sum to the
    /// channel count `N`, so the average channel has weight `1`. Keyed by channel.
    pub weights: BTreeMap<String, f64>,
    /// The effective number of independent channels, `1ᵀ R⁻¹ 1`: equal to `N` when the
    /// channels are uncorrelated, and fewer when they share a nuisance factor.
    pub effective_channels: f64,
}

/// Estimates each channel pair's redundancy correlation from a shared anchor, over the
/// items both channels scored.
///
/// This is one of the building blocks [`Fuser`](crate::Fuser) composes: it turns an
/// [`Anchor`] of representative queries into the pairwise correlations that
/// [`Fuser::refresh_coupling`](crate::Fuser::refresh_coupling) accumulates and
/// [`coupled_weights`] later discounts by.
///
/// The estimate is rank-based: Spearman's rank correlation over the both-scored items,
/// mapped to the Gaussian-copula linear correlation through `2·sin(π·ρ_s/6)` (Pearson's
/// exact relation between the two for a bivariate normal, so under a Gaussian
/// shared-nuisance model with loading `λ` the estimator recovers the same `λ²/(λ²+1)` a
/// linear correlation would). Ranks are the right basis here because Ruffle treats a
/// channel's scores as meaningful only up to an unknown monotone calibration: a linear
/// correlation on raw scores changes under a monotone rescaling (a compressed CLIP cone
/// against a heavy-tailed lexical score attenuates it), while the rank-based estimate is
/// invariant to any such rescaling and never compares one channel's raw magnitudes
/// against another's.
///
/// Each pair is correlated over only the items both channels scored. An absent score on
/// the anchor means the facet does not apply to that item, so restricting to the
/// both-scored items estimates redundancy over the population where both facets apply. A
/// pair whose overlap is below [`CouplingConfig::min_overlap`] is too thin for a stable
/// estimate and is omitted, as is a pair in which either channel has zero variance over
/// the overlap (the correlation is then undefined).
///
/// A random anchor is dominated by the bulk of irrelevant items, so this measures
/// within-bulk coupling. Using it as a stand-in for the coupling at the top of the pool
/// is a modelling assumption and is not corrected here.
#[must_use]
pub fn anchor_correlations(
    anchor: &Anchor,
    cfg: &CouplingConfig,
) -> BTreeMap<UnorderedPair<String>, PairObservation> {
    let mut out = BTreeMap::new();
    let n_channels = anchor.channels.len();
    for a in 0..n_channels {
        for b in (a + 1)..n_channels {
            // Both-scored restriction: keep only items where both channels scored.
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for (xi, yi) in anchor.scores[a].iter().zip(anchor.scores[b].iter()) {
                if let (Some(x), Some(y)) = (xi, yi) {
                    xs.push(*x);
                    ys.push(*y);
                }
            }
            let n_both = xs.len();
            if n_both < cfg.min_overlap {
                continue; // too thin for a stable estimate
            }
            if let Some(correlation) = copula_correlation(&xs, &ys) {
                let pair =
                    UnorderedPair::new(anchor.channels[a].clone(), anchor.channels[b].clone());
                out.insert(
                    pair,
                    PairObservation {
                        correlation,
                        n_both,
                    },
                );
            }
        }
    }
    out
}

/// The rank-based redundancy correlation: Spearman's rho mapped to the Gaussian-copula
/// linear correlation.
///
/// Spearman's rho is Pearson on midranks, so it is invariant to any strictly monotone
/// transform of either sample. The map `2·sin(π·ρ_s/6)` is the exact bivariate-normal
/// relation between Spearman's rho and the linear correlation, so the result estimates
/// the copula's linear correlation without ever trusting the samples' raw magnitudes.
/// It maps `[-1, 1]` onto `[-1, 1]` monotonically and fixes `0`. Returns `None` when the
/// underlying rank correlation is undefined (a constant sample, or a degenerate input).
fn copula_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let rho_s = pearson(&midranks(xs), &midranks(ys))?;
    let r = 2.0 * (std::f64::consts::FRAC_PI_6 * rho_s).sin();
    // rho_s is already in [-1, 1]; the bound here only absorbs rounding. min/max rather
    // than `clamp` so a pathological input degrades instead of panicking.
    #[allow(clippy::manual_clamp)]
    Some(r.min(1.0).max(-1.0))
}

/// Midranks of a sample: rank `1` = smallest, ties sharing the average of the ranks they
/// span. The standard rank transform behind Spearman's rho, well-defined even for
/// heavily tied samples such as integer-count scores.
fn midranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        // The tied run [i, j] shares the midrank (i + j)/2 + 1 (ranks are 1-based).
        let mut j = i;
        while j + 1 < n && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let mid = (i + j) as f64 / 2.0 + 1.0;
        for k in &idx[i..=j] {
            ranks[*k] = mid;
        }
        i = j + 1;
    }
    ranks
}

/// Pearson correlation of two equal-length samples, or `None` when it is undefined.
///
/// Returns `None` for an empty or length-mismatched input, for zero variance in either
/// sample (the correlation is then undefined), or for a non-finite result. The value is
/// clamped to `[-1, 1]` to absorb rounding past the bound.
fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len();
    if n == 0 || ys.len() != n {
        return None;
    }
    let nf = n as f64;
    let mean_x = xs.iter().sum::<f64>() / nf;
    let mean_y = ys.iter().sum::<f64>() / nf;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return None; // zero variance: correlation undefined
    }
    let r = cov / (var_x.sqrt() * var_y.sqrt());
    if r.is_finite() {
        Some(r.clamp(-1.0, 1.0))
    } else {
        None
    }
}

/// Turns per-channel discrimination weights into a redundancy-discounted set, so channels
/// that share a nuisance factor are not counted twice.
///
/// This is one of the building blocks [`Fuser`](crate::Fuser) composes. It takes the map
/// `g` of per-channel discrimination weights (from
/// [`discriminate`](crate::components::discriminate)) and the accumulated pairwise
/// `redundancy` correlations, and returns weights that discount channels which move
/// together.
///
/// The weights are `w = N · clamp₊(Σ̂⁻¹ 1) / ‖clamp₊(Σ̂⁻¹ 1)‖₁`, with `Σ̂ = D R D` and
/// `D = diag(1/√g_c)`, where `R` is the regularized redundancy correlation matrix:
///
/// 1. `R` has a unit diagonal. Its off-diagonal `[c, c']` is the pair's accumulated
///    redundancy mean, used only when coupling is enabled and the pair clears three
///    gates (otherwise `0`, meaning independence): its reliability (the accumulated
///    overlap count) is at least [`CouplingConfig::min_reliability`]; it is backed by at
///    least [`CouplingConfig::min_refreshes`] anchor refreshes (one refresh has zero
///    between-refresh variance by construction, so stability cannot be demonstrated
///    from it); and its variance across refreshes and strata is at most
///    [`CouplingConfig::stratum_stability_max_var`]. A redundancy that swings across
///    query strata would over-discount the independent regime, so an unstable pair
///    degrades to independence rather than suppressing a channel. Redundancy is
///    non-negative, so the value is capped to `[0, discount_cap]`: a negative
///    correlation is treated as independence rather than credited. `R` is then shrunk toward
///    the identity by [`CouplingConfig::shrink_to_identity`], which keeps it
///    positive-definite. With coupling off, or no pair clearing the gates, `R = I`
///    exactly.
/// 2. `D = diag(1/√g_c)`; every `g_c` is positive (discrimination floors it).
/// 3. `Σ̂ = D R D`. Solving `Σ̂ x = 1` falls back to independence (`x ∝ g`) if `Σ̂` is
///    somehow not positive-definite (it should be after the shrink).
/// 4. `x` is clamped at `0`, so a channel never cancels another, then renormalized to sum
///    to `N`; an all-zero clamp falls back to uniform weight `1`.
///
/// In the decoupled limit `R = I`, this gives `Σ̂ = diag(1/g_c)`, `Σ̂⁻¹ = diag(g_c)`, and
/// hence `w_c ∝ g_c`: with no redundancy, weight is proportional to discrimination.
/// `effective_channels = 1ᵀ R⁻¹ 1` is read from `R` alone, so it stays a channel count and
/// does not mix in `g`. Because the off-diagonals are non-negative, it lies in `(0, N]`
/// and never exceeds the channel count.
///
/// Coupling is off by default, and capped, shrunk, and gated when on, because the anchor
/// redundancy it consumes rests on two assumptions: that channel redundancy is
/// query-independent enough to amortize one pooled correlation, and that the bulk-stratum
/// coupling the anchor measures stands in for the coupling at the top of the pool. When
/// the evidence is thin, assuming independence is the recall-safe choice.
#[must_use]
pub fn coupled_weights(
    g: &BTreeMap<String, f64>,
    redundancy: &BTreeMap<UnorderedPair<String>, PairBaseline>,
    keys: &[String],
    cfg: &CouplingConfig,
) -> CoupledWeights {
    let n = keys.len();
    if n == 0 {
        return CoupledWeights {
            weights: BTreeMap::new(),
            effective_channels: 0.0,
        };
    }
    let nf = n as f64;

    // The discrimination vector, defended against a missing or non-positive entry. In
    // normal operation every key is present and floored positive (§4); a stray value
    // falls back to a neutral 1.0 rather than breaking the matrix.
    let gv: Vec<f64> = keys
        .iter()
        .map(|k| {
            let v = g.get(k).copied().unwrap_or(NEUTRAL_WEIGHT);
            if v.is_finite() && v > 0.0 {
                v
            } else {
                NEUTRAL_WEIGHT
            }
        })
        .collect();

    // Step 1: redundancy correlation R, gated and capped, then shrunk toward identity.
    // Bounds use min/max (never `clamp`) so an out-of-range knob, already rejected at
    // construction, still cannot panic here.
    let cap = cfg.discount_cap.max(0.0); // a negative cap means "no coupling"
    #[allow(clippy::manual_clamp)] // min/max never panics on an out-of-range knob
    let shrink = cfg.shrink_to_identity.min(1.0).max(0.0);
    let mut r = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        r[i][i] = 1.0;
    }
    if cfg.enabled {
        for i in 0..n {
            for j in (i + 1)..n {
                let pair = UnorderedPair::new(keys[i].clone(), keys[j].clone());
                let mut rho = 0.0;
                if let Some(red) = redundancy.get(&pair) {
                    // Three gates, each degrading to independence (0 = no discount), the
                    // recall-safe direction (§5.3, §5.4):
                    //   - reliability: the accumulated overlap count must clear the floor;
                    //   - refreshes: stability is a between-refresh property, so at least
                    //     min_refreshes anchor refreshes must back the pair (a single
                    //     refresh has zero between-refresh variance by construction and
                    //     demonstrates nothing);
                    //   - stratum stability: the cross-refresh/stratum variance of the
                    //     redundancy must be low enough. A correlation that swings across
                    //     query strata is not safe to amortize as one pooled discount, so
                    //     it is dropped rather than over-discounting the independent
                    //     regime and suppressing a facet (§5.3).
                    if red.redundancy.count() >= cfg.min_reliability
                        && red.refreshes >= cfg.min_refreshes
                        && red.redundancy.variance() <= cfg.stratum_stability_max_var
                    {
                        let mean = red.redundancy.mean();
                        if mean.is_finite() {
                            // Redundancy is the non-negative shared-nuisance term (§5.1):
                            // a negative anchor correlation clamps to 0 (independence),
                            // never a credit. Coupling only ever discounts.
                            rho = mean.min(cap).max(0.0);
                        }
                    }
                }
                r[i][j] = rho;
                r[j][i] = rho;
            }
        }
    }
    // Mandatory shrink toward the identity: R ← (1−s)·R + s·I. The unit diagonal is
    // preserved; off-diagonals are pulled toward zero, which also keeps R PD.
    for i in 0..n {
        for j in 0..n {
            let identity = if i == j { 1.0 } else { 0.0 };
            r[i][j] = (1.0 - shrink) * r[i][j] + shrink * identity;
        }
    }

    // Steps 2-3: Σ̂ = D R D with D = diag(1/√g_c); solve Σ̂ x = 1.
    let d_diag: Vec<f64> = gv.iter().map(|&gc| 1.0 / gc.sqrt()).collect();
    let mut sigma = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            sigma[i][j] = d_diag[i] * r[i][j] * d_diag[j];
        }
    }
    let ones = vec![1.0f64; n];
    // Fall back to independence (x ∝ g) if the solve fails (Σ̂ not PD; shouldn't happen).
    let x = solve_spd(&sigma, &ones).unwrap_or_else(|| gv.clone());

    // Step 4: clamp at 0 (never let a channel cancel another), renormalize to sum N.
    let clamped: Vec<f64> = x
        .iter()
        .map(|&xi| if xi.is_finite() { xi.max(0.0) } else { 0.0 })
        .collect();
    let sum: f64 = clamped.iter().sum();
    let weights_vec: Vec<f64> = if sum > 0.0 {
        clamped.iter().map(|&xi| nf * xi / sum).collect()
    } else {
        // Everything clamped to zero: fall back to the neutral weight for each channel.
        vec![NEUTRAL_WEIGHT; n]
    };
    let weights: BTreeMap<String, f64> = keys.iter().cloned().zip(weights_vec).collect();

    // Step 6: effective channels = 1ᵀ R⁻¹ 1, read from R alone (= N when R = I).
    let effective_channels = inverse_spd(&r)
        .map(|rinv| rinv.iter().flatten().sum::<f64>())
        .unwrap_or(nf);

    CoupledWeights {
        weights,
        effective_channels,
    }
}

/// The two set-overlap diagnostics returned by [`diagnostics`].
///
/// Both are dimensionless and read from set membership only. The fields are named so a
/// caller cannot silently swap the two. Marked `#[non_exhaustive]`: a result type that
/// callers read but never construct.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Diagnostics {
    /// The Jaccard overlap `|∩| / |∪|` of the top-`m` sets. High means the channels
    /// agree on which documents are relevant.
    pub confidence: f64,
    /// `1 − confidence`, gated on at least two sets. High means each channel is
    /// confident about different documents.
    pub conflict: f64,
}

/// Computes the two set-overlap diagnostics from top-set membership alone.
///
/// This is one of the building blocks [`Fuser`](crate::Fuser) composes for its
/// [`Fused::confidence`](crate::Fused::confidence) and
/// [`Fused::conflict`](crate::Fused::conflict). `top_sets` is each channel's top-`m`
/// candidate id list; the caller pre-filters to the discriminating channels, since the
/// diagnostics are only meaningful among channels that can rank. The result is a
/// [`Diagnostics`]:
///
/// - `confidence`: the Jaccard overlap `|∩| / |∪|` of the top-`m` sets. High means the
///   channels agree on which items are relevant.
/// - `conflict`: `1 − Jaccard`, gated on at least two sets. High means each channel is
///   confident about different items; since the channels were pre-filtered to the
///   discriminating ones, disjoint top sets are a genuine disagreement.
///
/// Fewer than two sets, or an empty union, yields `confidence` and `conflict` both `0.0`:
/// there is no overlap signal, so confidence is low and no conflict is asserted. The
/// computation reads set membership only; score values do not enter it.
#[must_use]
pub fn diagnostics<Id: std::hash::Hash + Eq + Clone>(
    top_sets: &[(String, Vec<Id>)],
) -> Diagnostics {
    if top_sets.len() < 2 {
        return Diagnostics {
            confidence: 0.0,
            conflict: 0.0,
        };
    }
    let sets: Vec<HashSet<Id>> = top_sets
        .iter()
        .map(|(_, ids)| ids.iter().cloned().collect())
        .collect();
    let mut union: HashSet<Id> = HashSet::new();
    for s in &sets {
        for id in s {
            union.insert(id.clone());
        }
    }
    if union.is_empty() {
        return Diagnostics {
            confidence: 0.0,
            conflict: 0.0,
        };
    }
    let intersection = union
        .iter()
        .filter(|id| sets.iter().all(|s| s.contains(*id)))
        .count();
    let jaccard = intersection as f64 / union.len() as f64;
    Diagnostics {
        confidence: jaccard,
        conflict: 1.0 - jaccard,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ChannelConfig;
    use crate::keys::ChannelId;
    use crate::score::{Direction, Score};
    use crate::summary::MeanVar;
    use approx::assert_abs_diff_eq;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Normal};

    /// A caller-side newtype: the only way a bare number becomes a [`Score`].
    struct Val(f64);
    impl Score for Val {
        fn value(&self) -> f64 {
            self.0
        }
    }

    fn key(s: &str) -> String {
        s.to_string()
    }

    fn channel(name: &str) -> ChannelConfig {
        ChannelConfig::new(ChannelId::new(name, "tag"), Direction::HigherIsBetter, None)
    }

    /// A `g` map from `(name, value)` pairs.
    fn gmap(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
        entries.iter().map(|(k, v)| (key(k), *v)).collect()
    }

    /// A reliable redundancy baseline with a given mean and reliability count, backed by
    /// enough refreshes to clear the between-refresh stability gate.
    fn redundancy(mean: f64, count: f64) -> PairBaseline {
        PairBaseline {
            redundancy: MeanVar::from_prior(mean, 0.0, count),
            refreshes: 2.0,
        }
    }

    // -- anchor_correlations -------------------------------------------------------

    #[test]
    fn recovers_injected_redundancy_on_unselected_sample() {
        // Two channels S_c = a·R + λ·Z + ε_c on a random (unselected) sample. The true
        // within-bulk correlation is λ²/(λ²+1); §5.2's collider is absent, so the
        // both-scored Pearson recovers it to ~2 digits. Checked for two loadings.
        for (lambda, target) in [(1.0_f64, 0.5_f64), (2.0, 0.8)] {
            let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE + lambda as u64);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let a = 3.0;
            let p = 1e-3; // rare relevance, so the sample is bulk-dominated
            let n = 20_000usize;

            let mut sa = Vec::with_capacity(n);
            let mut sb = Vec::with_capacity(n);
            for _ in 0..n {
                let relevant = if rng.gen_range(0.0_f64..1.0) < p {
                    1.0
                } else {
                    0.0
                };
                let z = normal.sample(&mut rng);
                let ea = normal.sample(&mut rng);
                let eb = normal.sample(&mut rng);
                sa.push(a * relevant + lambda * z + ea);
                sb.push(a * relevant + lambda * z + eb);
            }

            let cands: Vec<usize> = (0..n).collect();
            let ca = channel("a");
            let cb = channel("b");
            let anchor = Anchor::build(&cands, &[&ca, &cb], |id, k| {
                let v = if k == "a" { sa[*id] } else { sb[*id] };
                Some(Val(v))
            });

            let corr = anchor_correlations(&anchor, &CouplingConfig::default());
            let obs = corr.get(&UnorderedPair::new(key("a"), key("b"))).unwrap();
            assert_eq!(obs.n_both, n);
            assert_abs_diff_eq!(obs.correlation, target, epsilon = 0.03);
        }
    }

    #[test]
    fn redundancy_is_invariant_to_monotone_rescaling() {
        // The design premise (§2, §7) is that scores are meaningful only up to an
        // unknown monotone calibration, so the redundancy estimate must not move when a
        // channel's scores are monotonically rescaled. The rank-based estimator is
        // exactly invariant (identical ranks -> identical correlation); a linear
        // correlation on raw scores would attenuate under the exp warp below.
        let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let n = 5_000usize;
        let mut sa = Vec::with_capacity(n);
        let mut sb = Vec::with_capacity(n);
        for _ in 0..n {
            let z = normal.sample(&mut rng);
            sa.push(z + normal.sample(&mut rng));
            sb.push(z + normal.sample(&mut rng));
        }

        let cands: Vec<usize> = (0..n).collect();
        let ca = channel("a");
        let cb = channel("b");
        let raw = Anchor::build(&cands, &[&ca, &cb], |id, k| {
            Some(Val(if k == "a" { sa[*id] } else { sb[*id] }))
        });
        // The same channels with b's scores warped by the strictly monotone exp.
        let warped = Anchor::build(&cands, &[&ca, &cb], |id, k| {
            Some(Val(if k == "a" { sa[*id] } else { sb[*id].exp() }))
        });

        let cfg = CouplingConfig::default();
        let pair = UnorderedPair::new(key("a"), key("b"));
        let r_raw = anchor_correlations(&raw, &cfg)[&pair].correlation;
        let r_warped = anchor_correlations(&warped, &cfg)[&pair].correlation;
        assert!(
            r_raw > 0.3,
            "the shared nuisance must register, got {r_raw}"
        );
        assert_abs_diff_eq!(r_raw, r_warped, epsilon = 1e-12);
    }

    #[test]
    fn midranks_average_ties() {
        // [10, 20, 20, 30] -> ranks [1, 2.5, 2.5, 4]; order of appearance irrelevant.
        let r = midranks(&[20.0, 10.0, 30.0, 20.0]);
        assert_eq!(r, vec![2.5, 1.0, 4.0, 2.5]);
        // All tied: everyone at the midpoint (n+1)/2.
        assert_eq!(midranks(&[7.0, 7.0, 7.0]), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn both_scored_restriction_counts_only_shared_items() {
        // Channel b is absent (facet does not apply) on a known subset; the correlation
        // is computed over only the both-scored items and n_both reflects that.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let n = 200usize;
        let absent: HashSet<usize> = (0..n).filter(|i| i % 5 == 0).collect(); // 40 absent
        let expected_both = n - absent.len();

        let mut sa = Vec::with_capacity(n);
        let mut sb = Vec::with_capacity(n);
        for _ in 0..n {
            let z = normal.sample(&mut rng);
            sa.push(z + normal.sample(&mut rng) * 0.1);
            sb.push(z + normal.sample(&mut rng) * 0.1);
        }

        let cands: Vec<usize> = (0..n).collect();
        let ca = channel("a");
        let cb = channel("b");
        let anchor = Anchor::build(&cands, &[&ca, &cb], |id, k| {
            if k == "b" && absent.contains(id) {
                None
            } else if k == "a" {
                Some(Val(sa[*id]))
            } else {
                Some(Val(sb[*id]))
            }
        });

        let corr = anchor_correlations(&anchor, &CouplingConfig::default());
        let obs = corr.get(&UnorderedPair::new(key("a"), key("b"))).unwrap();
        assert_eq!(obs.n_both, expected_both);
        assert!(obs.correlation.is_finite() && obs.correlation > 0.5);
    }

    #[test]
    fn thin_overlap_is_omitted() {
        // Only three both-scored items, below the default min_overlap of 30.
        let n = 100usize;
        let present: HashSet<usize> = [0usize, 1, 2].into_iter().collect();
        let cands: Vec<usize> = (0..n).collect();
        let ca = channel("a");
        let cb = channel("b");
        let anchor = Anchor::build(&cands, &[&ca, &cb], |id, k| {
            if k == "b" && !present.contains(id) {
                None
            } else {
                Some(Val(*id as f64))
            }
        });
        let corr = anchor_correlations(&anchor, &CouplingConfig::default());
        assert!(corr.is_empty());
    }

    #[test]
    fn zero_variance_pair_is_omitted() {
        // Channel b is constant over the overlap: its variance is zero, so the
        // correlation is undefined and the pair is dropped.
        let n = 100usize;
        let cands: Vec<usize> = (0..n).collect();
        let ca = channel("a");
        let cb = channel("b");
        let anchor = Anchor::build(&cands, &[&ca, &cb], |id, k| {
            if k == "a" {
                Some(Val(*id as f64))
            } else {
                Some(Val(1.0)) // constant
            }
        });
        let corr = anchor_correlations(&anchor, &CouplingConfig::default());
        assert!(corr.is_empty());
    }

    #[test]
    fn overlap_exactly_at_min_is_kept() {
        // The thin-overlap drop is `n_both < min_overlap`: a pair whose overlap is
        // EXACTLY at the floor is reliable enough to keep (boundary `<`, not `<=`). With
        // `<=` the pair would be dropped at the threshold.
        let n = 30usize; // == default min_overlap
        let cands: Vec<usize> = (0..n).collect();
        let ca = channel("a");
        let cb = channel("b");
        let anchor = Anchor::build(&cands, &[&ca, &cb], |id, k| {
            // Both channels score every item with co-varying values, so the correlation
            // is defined (non-zero variance in each) and the only gate in play is overlap.
            let v = if k == "a" {
                *id as f64
            } else {
                *id as f64 * 2.0 + 1.0
            };
            Some(Val(v))
        });
        let cfg = CouplingConfig::default();
        assert_eq!(cfg.min_overlap, 30);
        let corr = anchor_correlations(&anchor, &cfg);
        let obs = corr
            .get(&UnorderedPair::new(key("a"), key("b")))
            .expect("a pair with overlap exactly at min_overlap must be kept");
        assert_eq!(obs.n_both, 30);
    }

    // -- pearson ------------------------------------------------------------------
    //
    // The `var_x <= 0.0 || var_y <= 0.0` guard (`||` -> `&&`) is an EQUIVALENT mutant:
    // a sum of squares is never negative, so `var <= 0` means `var == 0`, i.e. that
    // channel is constant over the overlap, which forces every deviation (and hence the
    // covariance) to zero. The correlation then evaluates to `0.0 / 0.0 = NaN`, which the
    // downstream `r.is_finite()` check converts to `None` -- the same result the guard
    // would have produced. No input distinguishes the two, so there is no test for it.

    #[test]
    fn pearson_rejects_length_mismatch() {
        // The input guard returns `None` for an empty OR length-mismatched sample (the
        // documented contract). A non-empty length mismatch is what distinguishes `||`
        // from `&&`: with `&&` the guard is false and the function proceeds to correlate
        // over the zipped prefix against a mean taken over the longer length, returning a
        // bogus `Some` instead of `None`.
        assert_eq!(pearson(&[0.0, 1.0, 2.0], &[10.0, 20.0]), None);
        assert_eq!(pearson(&[], &[]), None);
    }

    // -- coupled_weights -----------------------------------------------------------

    fn w(cw: &CoupledWeights, name: &str) -> f64 {
        *cw.weights.get(&key(name)).unwrap()
    }

    #[test]
    fn decoupled_limit_weights_proportional_to_g() {
        // R = I (coupling disabled): w_c ∝ g_c, normalized to sum N, all non-negative.
        let g = gmap(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let cfg = CouplingConfig::default(); // disabled
        let cw = coupled_weights(&g, &BTreeMap::new(), &keys, &cfg);
        assert_abs_diff_eq!(w(&cw, "a"), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&cw, "b"), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&cw, "c"), 1.5, epsilon = 1e-12);
        let total: f64 = cw.weights.values().sum();
        assert_abs_diff_eq!(total, 3.0, epsilon = 1e-12);
        assert!(cw.weights.values().all(|&x| x >= 0.0));
        assert_abs_diff_eq!(cw.effective_channels, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn disabled_coupling_is_exactly_the_decoupled_limit() {
        // A reliable redundant pair present, but coupling disabled: weights ignore it
        // and match w_c ∝ g_c exactly.
        let g = gmap(&[("a", 1.0), ("b", 1.0), ("c", 1.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let mut pairs = BTreeMap::new();
        pairs.insert(
            UnorderedPair::new(key("a"), key("b")),
            redundancy(0.8, 50.0),
        );
        let cfg = CouplingConfig {
            enabled: false,
            ..CouplingConfig::default()
        };
        let cw = coupled_weights(&g, &pairs, &keys, &cfg);
        for name in ["a", "b", "c"] {
            assert_abs_diff_eq!(w(&cw, name), 1.0, epsilon = 1e-12);
        }
        assert_abs_diff_eq!(cw.effective_channels, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn single_refresh_is_not_discounted() {
        // §5.3: stability across query strata is a between-refresh property, and one
        // refresh has zero between-refresh variance BY CONSTRUCTION, so it clears the
        // variance gate without demonstrating anything. The refresh gate keeps a
        // reliable-looking single-refresh pair at independence; a second agreeing
        // refresh (same mean, variance still 0) unlocks the discount.
        let g = gmap(&[("a", 1.0), ("b", 1.0), ("c", 1.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let cfg = CouplingConfig {
            enabled: true,
            ..CouplingConfig::default()
        };
        assert_eq!(cfg.min_refreshes, 2.0);

        let mut single = BTreeMap::new();
        single.insert(
            UnorderedPair::new(key("a"), key("b")),
            PairBaseline {
                redundancy: MeanVar::from_prior(0.8, 0.0, 50.0),
                refreshes: 1.0,
            },
        );
        let held = coupled_weights(&g, &single, &keys, &cfg);
        for name in ["a", "b", "c"] {
            assert_abs_diff_eq!(w(&held, name), 1.0, epsilon = 1e-12);
        }
        assert_abs_diff_eq!(held.effective_channels, 3.0, epsilon = 1e-12);

        // The same evidence over two refreshes discounts.
        let mut double = BTreeMap::new();
        double.insert(
            UnorderedPair::new(key("a"), key("b")),
            PairBaseline {
                redundancy: MeanVar::from_prior(0.8, 0.0, 50.0),
                refreshes: 2.0,
            },
        );
        let discounted = coupled_weights(&g, &double, &keys, &cfg);
        assert!(w(&discounted, "c") > w(&discounted, "a"));
        assert!(discounted.effective_channels < 3.0);
    }

    #[test]
    fn unreliable_pair_below_min_reliability_is_dropped() {
        // Coupling enabled but the pair's count is below min_reliability: no discount,
        // so weights stay at the decoupled limit.
        let g = gmap(&[("a", 1.0), ("b", 1.0)]);
        let keys = [key("a"), key("b")];
        let mut pairs = BTreeMap::new();
        // count 5 < default min_reliability 10
        pairs.insert(UnorderedPair::new(key("a"), key("b")), redundancy(0.8, 5.0));
        let cfg = CouplingConfig {
            enabled: true,
            ..CouplingConfig::default()
        };
        let cw = coupled_weights(&g, &pairs, &keys, &cfg);
        assert_abs_diff_eq!(w(&cw, "a"), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&cw, "b"), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cw.effective_channels, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn redundant_reliable_pair_is_downweighted_symmetrically() {
        // Channels a,b share a reliable redundancy; c is independent. a and b are
        // down-weighted equally and below c; weights sum to N and stay non-negative.
        let g = gmap(&[("a", 1.0), ("b", 1.0), ("c", 1.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let mut pairs = BTreeMap::new();
        pairs.insert(
            UnorderedPair::new(key("a"), key("b")),
            redundancy(0.8, 20.0),
        );
        let cfg = CouplingConfig {
            enabled: true,
            ..CouplingConfig::default()
        };
        let cw = coupled_weights(&g, &pairs, &keys, &cfg);
        // Symmetric: a and b carry the same weight.
        assert_abs_diff_eq!(w(&cw, "a"), w(&cw, "b"), epsilon = 1e-12);
        // The independent channel keeps more weight than the redundant pair.
        assert!(w(&cw, "c") > w(&cw, "a"));
        // Sums to N, all non-negative.
        let total: f64 = cw.weights.values().sum();
        assert_abs_diff_eq!(total, 3.0, epsilon = 1e-12);
        assert!(cw.weights.values().all(|&x| x >= 0.0));
        // Effective channels drop below N once two channels correlate.
        assert!(cw.effective_channels < 3.0);
        assert!(cw.effective_channels > 2.0);
    }

    #[test]
    fn negative_redundancy_is_treated_as_independence() {
        // §5.1: redundancy is the NON-NEGATIVE shared-nuisance term, and coupling is a
        // discount mechanism that never credits anti-correlation. A reliable pair whose
        // redundancy mean is negative is clamped to 0 (independence), so the weights match
        // the decoupled limit w_c ∝ g_c and effective_channels stays at N (never above it).
        let g = gmap(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let cfg = CouplingConfig {
            enabled: true,
            ..CouplingConfig::default()
        };

        let mut negative = BTreeMap::new();
        negative.insert(
            UnorderedPair::new(key("a"), key("b")),
            redundancy(-0.8, 50.0), // negative redundancy mean, otherwise reliable
        );
        let cw = coupled_weights(&g, &negative, &keys, &cfg);
        // Decoupled limit: w_c ∝ g_c normalized to sum N, exactly as if the pair were absent.
        assert_abs_diff_eq!(w(&cw, "a"), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&cw, "b"), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&cw, "c"), 1.5, epsilon = 1e-12);
        // No anti-correlation credit: effective channels stays at N, not above it.
        assert_abs_diff_eq!(cw.effective_channels, 3.0, epsilon = 1e-12);

        // A reliable POSITIVE pair of the same magnitude still discounts (existing behavior).
        let mut positive = BTreeMap::new();
        positive.insert(
            UnorderedPair::new(key("a"), key("b")),
            redundancy(0.8, 50.0),
        );
        let discounted = coupled_weights(&g, &positive, &keys, &cfg);
        assert!(discounted.effective_channels < 3.0);
        assert!(discounted.effective_channels > 2.0);
    }

    #[test]
    fn high_variance_reliable_pair_degrades_to_independence() {
        // §5.3 stratum-stability gate: a reliable pair whose redundancy is unstable
        // across query strata (high between-stratum variance) is dropped to
        // independence, while a low-variance reliable pair of the same mean/count is
        // used. The two pairs differ ONLY in variance, isolating the gate.
        let g = gmap(&[("a", 1.0), ("b", 1.0), ("c", 1.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let cfg = CouplingConfig {
            enabled: true,
            ..CouplingConfig::default()
        };

        // Low-variance reliable pair: the discount applies, so a and b drop below c.
        let mut stable = BTreeMap::new();
        stable.insert(
            UnorderedPair::new(key("a"), key("b")),
            PairBaseline {
                redundancy: MeanVar::from_prior(0.8, 0.0, 50.0),
                refreshes: 2.0,
            },
        );
        let used = coupled_weights(&g, &stable, &keys, &cfg);
        assert!(w(&used, "c") > w(&used, "a"));
        assert_abs_diff_eq!(w(&used, "a"), w(&used, "b"), epsilon = 1e-12);

        // High-variance reliable pair: variance 0.5 exceeds the default
        // stratum_stability_max_var of 0.25, so the discount is dropped and the weights
        // return to the independence limit w_c ∝ g_c (all equal here).
        let mut unstable = BTreeMap::new();
        unstable.insert(
            UnorderedPair::new(key("a"), key("b")),
            PairBaseline {
                redundancy: MeanVar::from_prior(0.8, 0.5, 50.0),
                refreshes: 2.0,
            },
        );
        let dropped = coupled_weights(&g, &unstable, &keys, &cfg);
        assert_abs_diff_eq!(w(&dropped, "a"), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&dropped, "b"), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&dropped, "c"), 1.0, epsilon = 1e-12);
        // No correlation used: effective channels back at N.
        assert_abs_diff_eq!(dropped.effective_channels, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn clamp_zeroes_a_negative_inverse_entry_and_renormalizes() {
        // A channel correlated with two others that are nearly independent of each other
        // makes its raw Σ̂⁻¹1 entry negative. With shrink 0 and a high cap, R is the raw
        // (still PD) correlation matrix and the negative entry clamps to 0.
        let g = gmap(&[("a", 1.0), ("b", 1.0), ("c", 1.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let mut pairs = BTreeMap::new();
        pairs.insert(
            UnorderedPair::new(key("a"), key("b")),
            redundancy(0.7, 50.0),
        );
        pairs.insert(
            UnorderedPair::new(key("a"), key("c")),
            redundancy(0.7, 50.0),
        );
        pairs.insert(
            UnorderedPair::new(key("b"), key("c")),
            redundancy(0.1, 50.0),
        );
        let cfg = CouplingConfig {
            enabled: true,
            discount_cap: 0.9,
            shrink_to_identity: 0.0,
            min_reliability: 1.0,
            ..CouplingConfig::default()
        };
        let cw = coupled_weights(&g, &pairs, &keys, &cfg);
        // Channel a is explained away: its weight clamps to exactly 0.
        assert_abs_diff_eq!(w(&cw, "a"), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&cw, "b"), 1.5, epsilon = 1e-9);
        assert_abs_diff_eq!(w(&cw, "c"), 1.5, epsilon = 1e-9);
        let total: f64 = cw.weights.values().sum();
        assert_abs_diff_eq!(total, 3.0, epsilon = 1e-12);
        assert!(cw.weights.values().all(|&x| x >= 0.0));
    }

    #[test]
    fn effective_channels_is_n_when_independent_and_fewer_when_correlated() {
        let keys = [key("a"), key("b")];
        let g = gmap(&[("a", 1.0), ("b", 1.0)]);

        // Independent (disabled): effective == N.
        let indep = coupled_weights(&g, &BTreeMap::new(), &keys, &CouplingConfig::default());
        assert_abs_diff_eq!(indep.effective_channels, 2.0, epsilon = 1e-12);

        // Strongly correlated: effective < N.
        let mut pairs = BTreeMap::new();
        pairs.insert(
            UnorderedPair::new(key("a"), key("b")),
            redundancy(0.9, 50.0),
        );
        let cfg = CouplingConfig {
            enabled: true,
            ..CouplingConfig::default()
        };
        let corr = coupled_weights(&g, &pairs, &keys, &cfg);
        assert!(corr.effective_channels < 2.0);
        assert!(corr.effective_channels > 0.0);
    }

    #[test]
    fn single_channel_carries_full_weight() {
        let g = gmap(&[("a", 2.0)]);
        let keys = [key("a")];
        let cw = coupled_weights(&g, &BTreeMap::new(), &keys, &CouplingConfig::default());
        assert_abs_diff_eq!(w(&cw, "a"), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cw.effective_channels, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn empty_channel_set_is_empty() {
        let cw = coupled_weights(
            &BTreeMap::new(),
            &BTreeMap::new(),
            &[],
            &CouplingConfig::default(),
        );
        assert!(cw.weights.is_empty());
        assert_eq!(cw.effective_channels, 0.0);
    }

    #[test]
    fn nonpositive_discrimination_is_floored_to_neutral() {
        // The discrimination vector defends each entry: a missing, zero, negative, or
        // non-finite `g` is replaced by a neutral 1.0 rather than fed into `D =
        // diag(1/sqrt(g))`, which a non-positive `g` would make non-finite and collapse
        // the channel's weight (§4). Coupling is disabled here, so `R = I` and a properly
        // floored, all-equal `g` gives every channel weight exactly 1.0.
        let keys = [key("a"), key("b")];
        let cfg = CouplingConfig::default(); // disabled, R = I

        // A ZERO entry pins the `> 0.0` boundary (`>` vs `>=`) and, together, the `&&`:
        // `0.0` is finite but not `> 0`, so it must floor to 1.0. A `>=` mutant would keep
        // `0.0` (-> 1/sqrt(0) = inf), and an `||` mutant would keep it via the finite arm;
        // either collapses channel a's weight to 0.
        let zero = coupled_weights(
            &gmap(&[("a", 0.0), ("b", 1.0)]),
            &BTreeMap::new(),
            &keys,
            &cfg,
        );
        assert_abs_diff_eq!(w(&zero, "a"), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&zero, "b"), 1.0, epsilon = 1e-12);

        // A NEGATIVE entry pins the `&&` (`||` would keep a finite-but-negative value via
        // its finite arm). It floors to 1.0; the mutant would keep -5.0 (-> 1/sqrt(-5) =
        // NaN) and zero the channel out.
        let neg = coupled_weights(
            &gmap(&[("a", -5.0), ("b", 1.0)]),
            &BTreeMap::new(),
            &keys,
            &cfg,
        );
        assert_abs_diff_eq!(w(&neg, "a"), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w(&neg, "b"), 1.0, epsilon = 1e-12);
    }

    // The final renormalization guard `if sum > 0.0` (`>` -> `>=`) is an EQUIVALENT
    // mutant. `sum` is the sum of the clamped solve `x = Σ̂⁻¹ 1`. For the SPD `Σ̂`,
    // `1ᵀΣ̂⁻¹1 > 0`, so at least one component of `x` is positive and the clamped sum is
    // strictly positive; the independence fallback `gv` is strictly positive too. `sum ==
    // 0.0` -- the only value at which `>` and `>=` differ -- is therefore unreachable, so
    // the uniform-fallback else branch never runs and no test can distinguish the two.

    // -- diagnostics ---------------------------------------------------------------

    #[test]
    fn diagnostics_identical_top_sets_are_confident_and_unconflicted() {
        let sets = vec![(key("a"), vec![1, 2, 3]), (key("b"), vec![1, 2, 3])];
        let Diagnostics {
            confidence,
            conflict,
        } = diagnostics(&sets);
        assert_abs_diff_eq!(confidence, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(conflict, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn diagnostics_disjoint_top_sets_conflict() {
        let sets = vec![(key("a"), vec![1, 2, 3]), (key("b"), vec![4, 5, 6])];
        let Diagnostics {
            confidence,
            conflict,
        } = diagnostics(&sets);
        assert_abs_diff_eq!(confidence, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(conflict, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn diagnostics_partial_overlap_is_jaccard() {
        // ∩ = {3,4} (2), ∪ = {1..6} (6) -> Jaccard 1/3.
        let sets = vec![(key("a"), vec![1, 2, 3, 4]), (key("b"), vec![3, 4, 5, 6])];
        let Diagnostics {
            confidence,
            conflict,
        } = diagnostics(&sets);
        assert_abs_diff_eq!(confidence, 1.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(conflict, 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn diagnostics_three_sets_generalize() {
        // ∩ = {3} (1), ∪ = {1,2,3,4,5} (5) -> Jaccard 0.2.
        let sets = vec![
            (key("a"), vec![1, 2, 3]),
            (key("b"), vec![2, 3, 4]),
            (key("c"), vec![3, 4, 5]),
        ];
        let Diagnostics {
            confidence,
            conflict,
        } = diagnostics(&sets);
        assert_abs_diff_eq!(confidence, 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(conflict, 0.8, epsilon = 1e-12);
    }

    #[test]
    fn diagnostics_need_two_sets() {
        let sets = vec![(key("a"), vec![1, 2, 3])];
        assert_eq!(
            diagnostics(&sets),
            Diagnostics {
                confidence: 0.0,
                conflict: 0.0,
            }
        );
        let none: Vec<(String, Vec<i32>)> = vec![];
        assert_eq!(
            diagnostics(&none),
            Diagnostics {
                confidence: 0.0,
                conflict: 0.0,
            }
        );
    }

    #[test]
    fn diagnostics_dedupe_within_a_set() {
        // Duplicates inside a list collapse: {1,2} vs {2} -> ∩={2}, ∪={1,2} -> 0.5.
        let sets = vec![(key("a"), vec![1, 1, 2, 2]), (key("b"), vec![2, 2, 2])];
        let confidence = diagnostics(&sets).confidence;
        assert_abs_diff_eq!(confidence, 0.5, epsilon = 1e-12);
    }
}
