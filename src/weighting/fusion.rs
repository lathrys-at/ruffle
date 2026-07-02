//! Weighted reciprocal-rank fusion.
//!
//! The fusion stays RRF, keeping its scale-freedom and bounded, rank-shaped per-channel
//! contribution, with per-channel weights added:
//!
//! ```text
//! score(d) = Σ_c  w_c / (η + rank_c(d))
//! ```
//!
//! An item absent from a channel's list is omitted from that channel's contribution
//! rather than charged a worst-rank penalty.
//!
//! # Rank derivation
//!
//! Each channel supplies an absolute rank per item, `1` = best. A `Scored`
//! channel is ranked by descending oriented score, so the highest score is rank `1`;
//! a `Ranks` channel is already in rank order, so list position `0` is rank `1`. Rank
//! is absolute rather than normalized by pool depth, so rank `3` contributes the same
//! whether the pool was `5` deep or `500`.
//!
//! # Ties and determinism
//!
//! `Id` is only `Hash + Eq + Clone`, so no ordering can be read off the ids
//! themselves, and a `HashMap` seed must never leak into the output. Every order is
//! imposed from a *first-seen index*: each distinct id is numbered by walking `obs` in
//! slice order, and within each channel in item order.
//!
//! Within a channel, tied scores share their *midrank*, the average of the ranks the
//! tie spans, so each tied item receives the identical contribution `w / (η + midrank)`
//! no matter how the tie was ordered in the input. A tie carries no ordering
//! information, so none is invented from input order; a channel's per-item
//! contributions are therefore invariant under any rank-equivalent reordering of its
//! list, ties included. The fused order is `(score DESC, first-seen index ASC)`: the
//! first-seen index breaks exact fused-score ties deterministically. The result is
//! identical across runs and independent of any hash seed, because the map is only
//! ever point-queried rather than iterated.

use crate::config::RrfConfig;
use crate::ingest::input::{ChannelInput, Items};
use crate::weighting::NEUTRAL_WEIGHT;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};

/// Fuses ranked channel outputs into one ranking by weighted reciprocal-rank fusion.
///
/// ```text
/// score(d) = Σ_c  w_c / (η + rank_c(d))
/// ```
///
/// Each item's fused score sums `w_c / (η + rank)` over the channels that ranked it. An
/// item a channel did not surface is omitted from that channel's sum rather than charged
/// a worst-rank penalty, because only present items are iterated. An item appearing in
/// several channels sums its per-channel contributions.
///
/// `weights` are the per-channel weights (normalized to sum to `N` upstream); a channel
/// present in `obs` but missing from `weights` defaults to the neutral weight (not
/// dropped), and a weight of `0.0` mutes the channel without erroring. `cfg.rrf_eta` is
/// the rank constant `η` and must be `>= 0` (conventionally around `60`); ranks start at
/// `1`, so a non-negative `η` keeps every `η + rank` denominator positive. A negative `η`
/// can drive `η + rank` to zero or below for the top ranks and produce non-finite or
/// negative contributions.
///
/// Precondition: each channel's item list must contain distinct ids. A repeated id within
/// one channel is counted once per occurrence (its contributions are double-counted),
/// so callers must not list an id twice in a single channel. (An id appearing across
/// different channels is the normal case and is summed as above; the precondition is per
/// channel only.)
///
/// Returns each surfaced item paired with its fused score, sorted best first by
/// `(score DESC, first-seen index ASC)`. Empty input yields an empty `Vec`.
#[must_use]
pub fn weighted_rrf<Id: std::hash::Hash + Eq + Clone>(
    obs: &[ChannelInput<Id>],
    weights: &BTreeMap<String, f64>,
    cfg: &RrfConfig,
) -> Vec<(Id, f64)> {
    let eta = cfg.rrf_eta;
    // Pass 1: number each distinct id by first appearance, walking channels in slice
    // order and items in list order. This index is the sole deterministic tiebreaker;
    // `order` holds the canonical owned id at each index.
    let mut first_seen: HashMap<Id, usize> = HashMap::new();
    let mut order: Vec<Id> = Vec::new();
    let mut note = |id: &Id| {
        let next = order.len();
        if let Entry::Vacant(slot) = first_seen.entry(id.clone()) {
            slot.insert(next);
            order.push(id.clone());
        }
    };
    for o in obs {
        match &o.items {
            Items::Scored(v) => v.iter().for_each(|(id, _)| note(id)),
            Items::Ranks(v) => v.iter().for_each(&mut note),
        }
    }

    // Pass 2: accumulate score(d) per first-seen index. Ranks are absolute (1 = best);
    // a missing weight defaults to the neutral 1.0.
    let mut scores: Vec<f64> = vec![0.0; order.len()];
    for o in obs {
        let w = weights.get(&o.key).copied().unwrap_or(NEUTRAL_WEIGHT);
        match &o.items {
            Items::Ranks(v) => {
                for (pos, id) in v.iter().enumerate() {
                    let rank = (pos + 1) as f64;
                    scores[first_seen[id]] += w / (eta + rank);
                }
            }
            Items::Scored(v) => {
                // Rank by descending oriented score; the first-seen tiebreak only fixes
                // the walk order (deterministic and hash-seed-free), never a tied item's
                // contribution.
                let mut positions: Vec<usize> = (0..v.len()).collect();
                positions.sort_by(|&a, &b| {
                    v[b].1
                        .partial_cmp(&v[a].1)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| first_seen[&v[a].0].cmp(&first_seen[&v[b].0]))
                });
                // Tied scores share their midrank: a tie carries no ordering
                // information, so each member gets the average of the ranks the tie
                // spans and the contribution is invariant to the tie's input order.
                let mut i = 0;
                while i < positions.len() {
                    let mut j = i;
                    while j + 1 < positions.len() && v[positions[j + 1]].1 == v[positions[i]].1 {
                        j += 1;
                    }
                    // Ranks are 1-based: the run spans ranks i+1 ..= j+1.
                    let midrank = (i + j) as f64 / 2.0 + 1.0;
                    for &p in &positions[i..=j] {
                        scores[first_seen[&v[p].0]] += w / (eta + midrank);
                    }
                    i = j + 1;
                }
            }
        }
    }

    // Pass 3: pair each id with its fused score and sort best first, breaking score
    // ties by ascending first-seen index. The enumerate index IS the first-seen index.
    let mut fused: Vec<(usize, Id, f64)> = order
        .into_iter()
        .enumerate()
        .map(|(i, id)| (i, id, scores[i]))
        .collect();
    fused.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    fused.into_iter().map(|(_, id, s)| (id, s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::input::Items;

    const EPS: f64 = 1e-12;

    fn scored(key: &str, v: Vec<(u32, f64)>) -> ChannelInput<u32> {
        ChannelInput {
            key: key.to_string(),
            items: Items::Scored(v),
        }
    }

    fn ranks(key: &str, v: Vec<u32>) -> ChannelInput<u32> {
        ChannelInput {
            key: key.to_string(),
            items: Items::Ranks(v),
        }
    }

    fn weights(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, w)| (k.to_string(), *w)).collect()
    }

    /// An [`RrfConfig`] with a chosen rank constant `η`, for the fusion calls below.
    fn rrf(eta: f64) -> RrfConfig {
        RrfConfig { rrf_eta: eta }
    }

    fn score_of(out: &[(u32, f64)], id: u32) -> Option<f64> {
        out.iter().find(|(i, _)| *i == id).map(|(_, s)| *s)
    }

    fn ids(out: &[(u32, f64)]) -> Vec<u32> {
        out.iter().map(|(i, _)| *i).collect()
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < EPS, "expected {b}, got {a}");
    }

    // A deterministic Fisher-Yates shuffle (fixed-seed LCG) so "rank-equivalent
    // reordering" tests are reproducible without a `rand` dependency.
    fn lcg_shuffle<T>(mut v: Vec<T>, mut seed: u64) -> Vec<T> {
        for i in (1..v.len()).rev() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (seed >> 33) as usize % (i + 1);
            v.swap(i, j);
        }
        v
    }

    // Two scored channels, eta = 1, neutral weights. Exact scores and order.
    //
    //   A: (1,0.9) (2,0.5) (3,0.1)  -> ranks 1,2,3
    //   B: (1,0.8) (2,0.4)          -> ranks 1,2
    //   item1 = 1/2 + 1/2 = 1.0 ; item2 = 1/3 + 1/3 ; item3 = 1/4 (A only)
    #[test]
    fn hand_case_exact_scores_and_order() {
        let obs = vec![
            scored("a", vec![(1, 0.9), (2, 0.5), (3, 0.1)]),
            scored("b", vec![(1, 0.8), (2, 0.4)]),
        ];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(1.0));
        approx(score_of(&out, 1).unwrap(), 1.0);
        approx(score_of(&out, 2).unwrap(), 1.0 / 3.0 + 1.0 / 3.0);
        approx(score_of(&out, 3).unwrap(), 0.25);
        assert_eq!(ids(&out), vec![1, 2, 3]);
    }

    // An item appearing in several channels sums its per-channel contributions.
    #[test]
    fn appears_in_several_channels_sums() {
        let obs = vec![
            ranks("a", vec![7]),
            ranks("b", vec![7]),
            ranks("c", vec![7]),
        ];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(1.0));
        // rank 1 in each of three channels: 3 * 1/(1+1).
        approx(score_of(&out, 7).unwrap(), 3.0 * 0.5);
    }

    // Absence omits, never penalizes: a strong single-channel item is not pushed below
    // items that merely appear in more channels at weaker ranks. `solo` keeps exactly
    // its one-channel mass and outranks `multi`, which spreads across three channels.
    #[test]
    fn absence_is_omission_not_penalty() {
        // solo (id 1) is rank 1 in A only; multi (id 100) is rank 6 in B, C, and D.
        // Distinct padding ids keep each channel's ranks well-defined.
        let obs = vec![
            ranks("a", vec![1]),
            ranks("b", vec![10, 11, 12, 13, 14, 100]),
            ranks("c", vec![20, 21, 22, 23, 24, 100]),
            ranks("d", vec![30, 31, 32, 33, 34, 100]),
        ];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(1.0));
        // solo (id 1): rank 1 in A only -> exactly 1/(1+1) = 0.5. No penalty added for
        // its absence from b/c/d.
        approx(score_of(&out, 1).unwrap(), 0.5);
        // multi (id 100): rank 6 in three channels -> 3 * 1/(1+6) = 3/7 ≈ 0.4286.
        approx(score_of(&out, 100).unwrap(), 3.0 / 7.0);
        // Despite appearing in three channels, multi does not outrank the absent-from-
        // three solo item.
        assert!(score_of(&out, 1).unwrap() > score_of(&out, 100).unwrap());
        let order = ids(&out);
        let pos1 = order.iter().position(|&x| x == 1).unwrap();
        let pos100 = order.iter().position(|&x| x == 100).unwrap();
        assert!(pos1 < pos100);
    }

    // Doubling a channel's weight scales its contribution by two.
    #[test]
    fn doubling_weight_scales_contribution() {
        let obs = vec![scored("a", vec![(1, 0.9)])];
        let base = weighted_rrf(&obs, &weights(&[("a", 1.0)]), &rrf(1.0));
        let doubled = weighted_rrf(&obs, &weights(&[("a", 2.0)]), &rrf(1.0));
        approx(score_of(&base, 1).unwrap(), 0.5);
        approx(score_of(&doubled, 1).unwrap(), 1.0);
        approx(
            score_of(&doubled, 1).unwrap(),
            2.0 * score_of(&base, 1).unwrap(),
        );
    }

    // Only the reweighted channel scales; the other channel's contribution is unchanged.
    #[test]
    fn reweighting_one_channel_leaves_others() {
        let obs = vec![ranks("a", vec![5]), ranks("b", vec![5])];
        let out = weighted_rrf(&obs, &weights(&[("a", 3.0)]), &rrf(1.0));
        // a contributes 3 * 1/2, b (missing weight -> 1.0) contributes 1/2.
        approx(score_of(&out, 5).unwrap(), 3.0 * 0.5 + 0.5);
    }

    // Equal weights reduce to standard (unweighted) RRF.
    #[test]
    fn equal_weights_reduce_to_standard_rrf() {
        let obs = vec![
            scored("a", vec![(1, 0.9), (2, 0.5)]),
            scored("b", vec![(2, 0.7), (1, 0.3)]),
        ];
        let eta = 60.0;
        let weighted = weighted_rrf(&obs, &weights(&[("a", 1.0), ("b", 1.0)]), &rrf(eta));
        // Standard RRF by hand: ranks are A:{1->1,2->2}, B:{2->1,1->2}.
        let std_1 = 1.0 / (eta + 1.0) + 1.0 / (eta + 2.0);
        let std_2 = 1.0 / (eta + 2.0) + 1.0 / (eta + 1.0);
        approx(score_of(&weighted, 1).unwrap(), std_1);
        approx(score_of(&weighted, 2).unwrap(), std_2);
        // Neutral default (empty map) matches explicit equal weights.
        let neutral = weighted_rrf(&obs, &BTreeMap::new(), &rrf(eta));
        assert_eq!(weighted, neutral);
    }

    // Absolute rank is depth-robust: an item at rank 3 contributes the same in a pool
    // of 5 and a pool of 500.
    #[test]
    fn depth_robust_absolute_rank() {
        let eta = 60.0;
        let shallow: Vec<u32> = vec![1, 2, 999, 4, 5]; // 999 at index 2 -> rank 3
        let mut deep: Vec<u32> = vec![1, 2, 999];
        deep.extend(1000..1497); // pad to depth 500, 999 still at rank 3
        assert_eq!(deep.len(), 500);

        let out_shallow = weighted_rrf(&[ranks("a", shallow)], &BTreeMap::new(), &rrf(eta));
        let out_deep = weighted_rrf(&[ranks("a", deep)], &BTreeMap::new(), &rrf(eta));

        let expected = 1.0 / (eta + 3.0);
        approx(score_of(&out_shallow, 999).unwrap(), expected);
        approx(score_of(&out_deep, 999).unwrap(), expected);
        approx(
            score_of(&out_shallow, 999).unwrap(),
            score_of(&out_deep, 999).unwrap(),
        );
    }

    // Rank-equivalent reorderings of a scored list (distinct scores -> same rank order)
    // produce an identical fused result.
    #[test]
    fn determinism_under_rank_equivalent_reordering() {
        let base = scored("a", vec![(1, 0.9), (2, 0.7), (3, 0.5), (4, 0.3), (5, 0.1)]);
        let canonical = weighted_rrf(std::slice::from_ref(&base), &BTreeMap::new(), &rrf(30.0));
        let original = match &base.items {
            Items::Scored(v) => v.clone(),
            Items::Ranks(_) => unreachable!(),
        };
        for seed in [1u64, 42, 12345, 9_999_999] {
            let shuffled = lcg_shuffle(original.clone(), seed);
            let obs = vec![scored("a", shuffled)];
            let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(30.0));
            assert_eq!(out, canonical, "reordering with seed {seed} changed output");
        }
    }

    // Running twice yields a byte-identical Vec, independent of the per-call HashMap
    // seed. Includes score ties so the tiebreak path is exercised.
    #[test]
    fn determinism_two_runs_identical() {
        let obs = vec![
            scored("a", vec![(10, 0.5), (20, 0.5), (30, 0.5)]), // all tied
            ranks("b", vec![30, 10, 20]),
        ];
        let w = weights(&[("a", 1.5), ("b", 0.5)]);
        let first = weighted_rrf(&obs, &w, &rrf(7.0));
        let second = weighted_rrf(&obs, &w, &rrf(7.0));
        assert_eq!(first, second);
    }

    // Within a channel, tied scores share their midrank: both members of a two-way tie
    // at the top take rank 1.5 and identical contributions; the fused order between
    // them falls to the first-seen tiebreak.
    #[test]
    fn within_channel_ties_share_their_midrank() {
        // Both score 0.5: ranks 1 and 2 average to 1.5, contribution 1/(1+1.5) each.
        let obs = vec![scored("a", vec![(100, 0.5), (200, 0.5)])];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(1.0));
        approx(score_of(&out, 100).unwrap(), 1.0 / 2.5);
        approx(score_of(&out, 200).unwrap(), 1.0 / 2.5);
        // Equal fused scores: first-seen order decides the output order.
        assert_eq!(ids(&out), vec![100, 200]);

        // A three-way tie below a distinct top: top takes rank 1, the tie spans ranks
        // 2..4 and shares midrank 3.
        let obs = vec![scored("a", vec![(1, 0.9), (10, 0.5), (20, 0.5), (30, 0.5)])];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(1.0));
        approx(score_of(&out, 1).unwrap(), 1.0 / 2.0);
        for id in [10, 20, 30] {
            approx(score_of(&out, id).unwrap(), 1.0 / 4.0);
        }
    }

    // A tie carries no ordering information, so reordering tied items must not move any
    // item's fused score. (Under sequential tie-ranking, the earlier-listed item of a
    // tie took the better rank and input order leaked into the scores.)
    #[test]
    fn tied_scores_are_order_invariant() {
        let base = vec![(1u32, 0.9), (10, 0.5), (20, 0.5), (30, 0.5), (2, 0.1)];
        let canonical = weighted_rrf(&[scored("a", base.clone())], &BTreeMap::new(), &rrf(30.0));
        for seed in [3u64, 77, 4242] {
            let shuffled = lcg_shuffle(base.clone(), seed);
            let out = weighted_rrf(&[scored("a", shuffled)], &BTreeMap::new(), &rrf(30.0));
            // Per-item fused scores are identical under any reordering; only the
            // presentation order of exactly-tied fused scores may follow first-seen.
            for (id, s) in &canonical {
                approx(score_of(&out, *id).unwrap(), *s);
            }
        }
    }

    // A ranks-only channel: the given order is the rank.
    #[test]
    fn ranks_only_channel_uses_position_as_rank() {
        let obs = vec![ranks("a", vec![3, 1, 2])];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(1.0));
        approx(score_of(&out, 3).unwrap(), 1.0 / 2.0); // pos 0 -> rank 1
        approx(score_of(&out, 1).unwrap(), 1.0 / 3.0); // pos 1 -> rank 2
        approx(score_of(&out, 2).unwrap(), 1.0 / 4.0); // pos 2 -> rank 3
        assert_eq!(ids(&out), vec![3, 1, 2]);
    }

    #[test]
    fn empty_obs_yields_empty() {
        let obs: Vec<ChannelInput<u32>> = vec![];
        assert!(weighted_rrf(&obs, &BTreeMap::new(), &rrf(60.0)).is_empty());
    }

    #[test]
    fn empty_channels_contribute_nothing() {
        // Only empty channels -> empty output.
        let only_empty = vec![scored("a", vec![]), ranks("b", vec![])];
        assert!(weighted_rrf(&only_empty, &BTreeMap::new(), &rrf(60.0)).is_empty());

        // An empty channel alongside a populated one leaves the populated result intact.
        let mixed = vec![scored("a", vec![]), ranks("b", vec![1, 2])];
        let out = weighted_rrf(&mixed, &BTreeMap::new(), &rrf(1.0));
        approx(score_of(&out, 1).unwrap(), 0.5);
        approx(score_of(&out, 2).unwrap(), 1.0 / 3.0);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn single_item() {
        let out = weighted_rrf(&[ranks("a", vec![42])], &BTreeMap::new(), &rrf(60.0));
        assert_eq!(out.len(), 1);
        approx(score_of(&out, 42).unwrap(), 1.0 / 61.0);
    }

    // A channel present in obs but missing from the weight map defaults to 1.0.
    #[test]
    fn missing_weight_defaults_to_one() {
        let obs = vec![ranks("a", vec![1]), ranks("b", vec![2])];
        // Only "a" is weighted; "b" must default to 1.0, not be dropped.
        let out = weighted_rrf(&obs, &weights(&[("a", 1.0)]), &rrf(1.0));
        approx(score_of(&out, 1).unwrap(), 0.5);
        approx(score_of(&out, 2).unwrap(), 0.5);
        // Identical to giving both explicit 1.0.
        let explicit = weighted_rrf(&obs, &weights(&[("a", 1.0), ("b", 1.0)]), &rrf(1.0));
        assert_eq!(out, explicit);
    }

    // A weight of 0.0 mutes a channel without erroring; its items contribute zero.
    #[test]
    fn zero_weight_mutes_channel() {
        let obs = vec![ranks("a", vec![1]), ranks("b", vec![1, 2])];
        let out = weighted_rrf(&obs, &weights(&[("a", 0.0), ("b", 1.0)]), &rrf(1.0));
        // id 1: muted A contributes 0, B rank 1 contributes 1/2.
        approx(score_of(&out, 1).unwrap(), 0.5);
        // id 2: B rank 2 only.
        approx(score_of(&out, 2).unwrap(), 1.0 / 3.0);

        // An item present ONLY in a muted channel still appears, with score 0.0.
        let solo_muted = vec![ranks("a", vec![1]), ranks("b", vec![2])];
        let out2 = weighted_rrf(&solo_muted, &weights(&[("a", 0.0), ("b", 1.0)]), &rrf(1.0));
        approx(score_of(&out2, 1).unwrap(), 0.0);
        approx(score_of(&out2, 2).unwrap(), 0.5);
        // Zero-scored item sorts last (score DESC).
        assert_eq!(ids(&out2), vec![2, 1]);
    }

    // A mixed scored + ranks channel set with non-trivial weights, exact arithmetic.
    #[test]
    fn mixed_scored_and_ranks_with_weights() {
        let obs = vec![
            scored("a", vec![(1, 0.9), (2, 0.1)]), // ranks 1,2
            ranks("b", vec![2, 1]),                // 2->rank1, 1->rank2
        ];
        let w = weights(&[("a", 2.0), ("b", 1.0)]);
        let out = weighted_rrf(&obs, &w, &rrf(1.0));
        // id1: a w=2 rank1 (2*1/2=1.0) + b rank2 (1/3)
        approx(score_of(&out, 1).unwrap(), 1.0 + 1.0 / 3.0);
        // id2: a w=2 rank2 (2*1/3=2/3) + b rank1 (1/2)
        approx(score_of(&out, 2).unwrap(), 2.0 / 3.0 + 0.5);
        assert_eq!(ids(&out), vec![1, 2]);
    }
}
