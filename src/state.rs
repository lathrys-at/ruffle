//! Persistent state: the single mergeable object.
//!
//! Everything Ruffle persists is a confidence-weighted summary plus the identifiers
//! needed to merge it safely. This module defines the state types, their canonical
//! (`BTreeMap`-ordered) serialization, and the one reconciliation operation that serves
//! as streaming update, operator prior, and cross-deployment merge at once. The merge
//! gates on format version, statistic fingerprint, and the required per-channel tag,
//! reports an advisory divergence alongside, and is paired with the safe [`rekey`](RuffleState::rekey)
//! rename and the flagged [`decay`](RuffleState::decay) of confidence.

use crate::error::Mismatch;
use crate::keys::{StatFingerprint, UnorderedPair};
use crate::score::Direction;
use crate::summary::MeanVar;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The finite ceiling on a per-channel divergence.
///
/// A standardized distance is unbounded when the pooled spread collapses to zero while
/// the means still differ (an infinite z-distance). The metric caps at this value so
/// the advisory number stays finite and serializes cleanly, while still reading as
/// "very large" against the single-digit distances ordinary drift produces.
const DIVERGENCE_CAP: f64 = 1.0e6;

/// The persistent statistics Ruffle keeps for one channel: the baseline that
/// standardizes how well the channel separates its top results from the rest, the
/// reference used to judge absolute score quality, and the model-version tag that gates
/// merging.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelSummary {
    /// The baseline for this channel's separation between top and bulk scores. Each
    /// query yields a raw separation reading; standardizing it against this running
    /// mean and variance turns it into a scale-free measure of how well the channel
    /// discriminates.
    pub separation: MeanVar,
    /// The reference for what a good score looks like on this channel, either seeded
    /// from a declared [`GoodScore`](crate::score::GoodScore) through
    /// [`MeanVar::from_prior`] or learned from observed top scores. A query's top score
    /// is graded against it with [`MeanVar::zscore`] to gauge absolute quality.
    pub reference: MeanVar,
    /// The model-version tag identifying which model produced this channel's scores.
    /// [`RuffleState::merge`] checks it for equality on any channel shared between two
    /// states and refuses on a mismatch, so accumulated statistics can never silently
    /// span a model swap. The check runs at merge time rather than on write: building
    /// the `channels` map or setting a `tag` by hand is fine, and the gate catches any
    /// conflict when two states are combined.
    pub tag: String,
}

impl ChannelSummary {
    /// An empty summary carrying only the required tag. The separation baseline and
    /// reference start empty and accumulate from traffic or a declared prior.
    pub fn new(tag: String) -> Self {
        Self {
            separation: MeanVar::new(),
            reference: MeanVar::new(),
            tag,
        }
    }

    /// A summary with a pre-seeded good-score reference (e.g. from a declared
    /// [`GoodScore`](crate::score::GoodScore)).
    pub fn with_reference(tag: String, reference: MeanVar) -> Self {
        Self {
            separation: MeanVar::new(),
            reference,
            tag,
        }
    }
}

/// The persistent statistics Ruffle keeps for one pair of channels: their accumulated
/// redundancy correlation plus how many anchor refreshes back it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairSummary {
    /// The accumulated redundancy correlation between the two channels, measured on the
    /// shared anchor. Each refresh merges in one reading weighted by its both-scored
    /// overlap, so the mean is the overlap-weighted point estimate, the count is the
    /// total overlap backing it (the reliability gate), and the variance is the
    /// overlap-weighted spread across refreshes and strata (the stability gate).
    pub redundancy: MeanVar,
    /// How many anchor refreshes contributed to `redundancy`. Stability across query
    /// strata is a between-refresh property, so a discount is applied only once at
    /// least [`CouplingConfig::min_refreshes`](crate::config::CouplingConfig::min_refreshes)
    /// refreshes agree; a single refresh has zero between-refresh variance by
    /// construction and carries no evidence of stability. Fractional to support decay.
    #[serde(default)]
    pub refreshes: f64,
}

impl PairSummary {
    /// An empty pair summary.
    pub fn new() -> Self {
        Self {
            redundancy: MeanVar::new(),
            refreshes: 0.0,
        }
    }
}

impl Default for PairSummary {
    fn default() -> Self {
        Self::new()
    }
}

/// The persistent statistics Ruffle accumulates: a confidence-weighted summary per
/// channel and per channel pair, plus the versioning needed to merge two of them safely.
///
/// Maps are stored ordered (`BTreeMap`), so two states with identical contents serialize
/// byte-for-byte identically. That makes a serialized state content-addressable and its
/// diffs clean.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuffleState {
    /// Schema version, checked to decide whether this build can parse this file.
    ///
    /// Library-managed: set by [`RuffleState::new`], by deserialization, and by
    /// [`merge`](Self::merge), and not by direct downstream mutation. A downstream caller
    /// editing it in place would bypass the merge compatibility gate, so the field is
    /// crate-private and readable through [`format_version`](Self::format_version).
    pub(crate) format_version: u32,
    /// Statistic definitions plus per-channel orientation, checked to decide whether two
    /// states measured the same thing the same way.
    ///
    /// Library-managed: set by [`RuffleState::new`], by deserialization, and by
    /// [`merge`](Self::merge), and not by direct downstream mutation. A downstream caller
    /// editing it in place would bypass the merge compatibility gate, so the field is
    /// crate-private and readable through [`fingerprint`](Self::fingerprint).
    pub(crate) fingerprint: StatFingerprint,
    /// Per-channel summaries, keyed by the stable join handle.
    pub channels: BTreeMap<String, ChannelSummary>,
    /// Per-pair coupling summaries, keyed by the canonical unordered channel pair.
    ///
    /// Serialized as an array of `[pair, summary]` entries: an `UnorderedPair` is a
    /// two-field tuple, which JSON cannot use as an object key, so the map is written as
    /// a (canonically ordered) sequence instead, through the crate-internal
    /// `pairs_as_seq` serde adapter.
    #[serde(with = "pairs_as_seq")]
    pub pairs: BTreeMap<UnorderedPair<String>, PairSummary>,
}

/// An advisory standardized distance between two states' per-channel summaries.
///
/// The number is purely advisory and never gates a merge; gating is done by the
/// model-version tag. It flags a silent model swap, where two summaries have drifted
/// far apart while their model-version tags still match, so a caller can catch it at
/// the reconcile boundary.
/// Marked `#[non_exhaustive]`: a result type produced by [`RuffleState::divergence`]
/// and [`RuffleState::merge`] that callers read but never construct.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Divergence {
    /// Per-channel standardized distance between the two states.
    pub per_channel: BTreeMap<String, f64>,
    /// The largest per-channel distance: the single summary number a caller can
    /// threshold on.
    pub max: f64,
}

/// How [`RuffleState::merge`] treats incompatible inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MergePolicy {
    /// Refuses on any format, fingerprint, or tag mismatch. The only policy for now;
    /// the enum leaves room for future, looser policies.
    #[default]
    Strict,
}

impl RuffleState {
    /// The format version this build writes.
    ///
    /// Version history:
    /// - `1`: initial schema.
    /// - `2`: [`PairSummary`] gains the `refreshes` count backing the coupling
    ///   stability gate.
    pub const FORMAT_VERSION: u32 = 2;

    /// An empty state at the current format version with the given fingerprint.
    pub fn new(fingerprint: StatFingerprint) -> Self {
        Self {
            format_version: Self::FORMAT_VERSION,
            fingerprint,
            channels: BTreeMap::new(),
            pairs: BTreeMap::new(),
        }
    }

    /// The schema format version this state was built or loaded at.
    ///
    /// The field is library-managed and read-only to callers: it is set by
    /// [`new`](Self::new), by deserialization, and by [`merge`](Self::merge). Keeping it
    /// out of reach means a caller cannot edit it in place and slip a mismatched state
    /// past the merge compatibility check.
    #[must_use]
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// The statistic fingerprint this state was built or loaded with.
    ///
    /// The fingerprint records which statistic definition and per-channel orientation the
    /// state was measured under, letting a merge check whether two states measured the
    /// same thing the same way. It is library-managed and read-only to callers, set by
    /// [`new`](Self::new), by deserialization, and by [`merge`](Self::merge), so a caller
    /// cannot edit it in place and slip an incompatible state past the merge check.
    #[must_use]
    pub fn fingerprint(&self) -> &StatFingerprint {
        &self.fingerprint
    }

    /// Combines several states into one, returning the merged state and an advisory
    /// divergence between the inputs.
    ///
    /// The same operation serves as a streaming update, an operator prior, and
    /// cross-deployment reconciliation: a live update merges a count-1 state, a prior
    /// merges a hand-authored one, and reconciliation merges several saved states. It is
    /// associative and commutative and, with decay off, exact up to f64 rounding, because
    /// every persisted quantity is a [`MeanVar`] and [`MeanVar::merge`] has those
    /// properties.
    ///
    /// Under [`MergePolicy::Strict`] it refuses on the first incompatibility, checked in
    /// this order:
    ///
    /// 1. `format_version` must equal this build's [`FORMAT_VERSION`](Self::FORMAT_VERSION)
    ///    and agree across all parts, and the fingerprint's `stat_version` must equal
    ///    this build's [`StatFingerprint::STAT_VERSION`] ([`Mismatch::FormatVersion`] /
    ///    [`Mismatch::Fingerprint`]). A state that parses under an older definition still
    ///    measures different statistics, so successful parsing does not imply
    ///    compatibility.
    /// 2. The fingerprint must be compatible: equal statistic version and baseline mode
    ///    across all parts ([`Mismatch::Fingerprint`]), and no channel present in more
    ///    than one part may carry a conflicting orientation
    ///    ([`Mismatch::DirectionConflict`]). The per-channel orientation maps are
    ///    *unioned* rather than required to be equal, so a part that introduces a new
    ///    channel does not block the merge.
    /// 3. A channel present in more than one part must carry the same model-version tag
    ///    in each ([`Mismatch::Tag`]). A model swap under a reused key would corrupt the
    ///    accumulated statistics, so it is refused.
    ///
    /// On success, channels and pairs are unioned by key, and any key shared across parts
    /// has its summaries merged with [`MeanVar::merge`]. An empty `parts` slice returns
    /// [`Mismatch::Empty`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ruffle::{BaselineMode, Direction, MergePolicy, RuffleState, StatFingerprint};
    /// use std::collections::BTreeMap;
    ///
    /// let mut dirs = BTreeMap::new();
    /// dirs.insert(String::from("lexical"), Direction::HigherIsBetter);
    /// let a = RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, dirs.clone()));
    /// let b = RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, dirs));
    ///
    /// // Compatible states reconcile; the divergence is advisory and never gates.
    /// let (merged, divergence) = RuffleState::merge(&[&a, &b], MergePolicy::Strict)?;
    /// assert_eq!(merged.format_version(), RuffleState::FORMAT_VERSION);
    /// assert_eq!(divergence.max, 0.0);
    /// # Ok::<(), ruffle::Mismatch>(())
    /// ```
    pub fn merge(
        parts: &[&RuffleState],
        policy: MergePolicy,
    ) -> Result<(RuffleState, Divergence), Mismatch> {
        // Strict is the only policy today; future, looser policies branch here.
        match policy {
            MergePolicy::Strict => {}
        }

        let (first, rest) = parts.split_first().ok_or(Mismatch::Empty)?;

        // Gate 1: this build's schema and statistic versions, then one schema version
        // across all parts. A state at another version may still parse, but its
        // summaries measure different statistics, so it is refused rather than blended.
        if first.format_version != Self::FORMAT_VERSION {
            return Err(Mismatch::FormatVersion {
                left: Self::FORMAT_VERSION,
                right: first.format_version,
            });
        }
        if first.fingerprint.stat_version != StatFingerprint::STAT_VERSION {
            return Err(Mismatch::Fingerprint);
        }
        for p in rest {
            if p.format_version != first.format_version {
                return Err(Mismatch::FormatVersion {
                    left: first.format_version,
                    right: p.format_version,
                });
            }
        }

        // Gate 2a: one statistic definition and baseline mode across all parts.
        for p in rest {
            if p.fingerprint.stat_version != first.fingerprint.stat_version
                || p.fingerprint.baseline_mode != first.fingerprint.baseline_mode
            {
                return Err(Mismatch::Fingerprint);
            }
        }

        // Gate 2b: union the orientation maps, refusing on a per-channel conflict. A
        // channel new to one part is added; a channel in several must agree.
        let mut directions: BTreeMap<String, Direction> = BTreeMap::new();
        for p in parts {
            for (key, dir) in &p.fingerprint.directions {
                match directions.get(key) {
                    Some(existing) if existing != dir => {
                        return Err(Mismatch::DirectionConflict {
                            channel: key.to_string(),
                        });
                    }
                    _ => {
                        directions.insert(key.clone(), *dir);
                    }
                }
            }
        }

        // Gate 3 + channel union: fold each part's channels in, checking the required
        // tag on every shared key before merging its summaries.
        let mut channels: BTreeMap<String, ChannelSummary> = BTreeMap::new();
        for p in parts {
            for (key, summary) in &p.channels {
                match channels.get_mut(key) {
                    Some(existing) => {
                        if existing.tag != summary.tag {
                            return Err(Mismatch::Tag {
                                channel: key.to_string(),
                                left: existing.tag.to_string(),
                                right: summary.tag.to_string(),
                            });
                        }
                        existing.separation.merge_in(&summary.separation);
                        existing.reference.merge_in(&summary.reference);
                    }
                    None => {
                        channels.insert(key.clone(), summary.clone());
                    }
                }
            }
        }

        // Pair union: a shared pair's redundancy summaries merge and its refresh
        // counts add, so the stability gate sees the pooled between-refresh evidence.
        let mut pairs: BTreeMap<UnorderedPair<String>, PairSummary> = BTreeMap::new();
        for p in parts {
            for (pair, summary) in &p.pairs {
                match pairs.get_mut(pair) {
                    Some(existing) => {
                        existing.redundancy.merge_in(&summary.redundancy);
                        existing.refreshes += summary.refreshes;
                    }
                    None => {
                        pairs.insert(pair.clone(), summary.clone());
                    }
                }
            }
        }

        let merged = RuffleState {
            format_version: first.format_version,
            fingerprint: StatFingerprint {
                stat_version: first.fingerprint.stat_version,
                baseline_mode: first.fingerprint.baseline_mode,
                directions,
            },
            channels,
            pairs,
        };

        // Advisory divergence across the inputs: for each channel, the largest distance
        // seen between any two parts that both carry it (§8).
        let mut per_channel: BTreeMap<String, f64> = BTreeMap::new();
        for (i, a) in parts.iter().enumerate() {
            for b in &parts[i + 1..] {
                for (key, d) in a.divergence(b).per_channel {
                    let slot = per_channel.entry(key).or_insert(0.0);
                    if d > *slot {
                        *slot = d;
                    }
                }
            }
        }
        let max = per_channel.values().copied().fold(0.0_f64, f64::max);

        Ok((merged, Divergence { per_channel, max }))
    }

    /// The advisory divergence between this state and another, callable on its own before
    /// any merge.
    ///
    /// For every channel present in *both* states it reports a standardized distance,
    /// `|mean_a − mean_b| / pooled_std` with `pooled_std = sqrt((var_a + var_b) / 2)`,
    /// computed over each of the channel's two baselines, and keeps the larger:
    ///
    /// - the separation baseline, which shifts when the channel's ranking behaviour
    ///   changes shape; and
    /// - the good-score reference, which lives in the channel's native units and so is
    ///   the one that jumps under a silent model swap. The separation statistic is
    ///   deliberately scale- and shift-invariant, so a swap that rescales scores can
    ///   leave it untouched; the reference is where that swap shows.
    ///
    /// When a baseline pair's pooled spread is zero or non-finite its distance is
    /// undefined: it contributes `0.0` if the means also coincide, and the finite
    /// ceiling `1e6` otherwise (an infinite z-distance, clamped). Channels present in
    /// only one state are absent from the result. The number is purely advisory and
    /// never gates a merge.
    #[must_use]
    pub fn divergence(&self, other: &RuffleState) -> Divergence {
        let mut per_channel: BTreeMap<String, f64> = BTreeMap::new();
        let mut max = 0.0_f64;
        for (key, a) in &self.channels {
            if let Some(b) = other.channels.get(key) {
                let d = standardized_distance(&a.separation, &b.separation)
                    .max(standardized_distance(&a.reference, &b.reference));
                if d > max {
                    max = d;
                }
                per_channel.insert(key.clone(), d);
            }
        }
        Divergence { per_channel, max }
    }

    /// Renames a channel's key from `from` to `to`, moving all of its statistics with it.
    ///
    /// Everything keyed by `from` moves to `to`: the channel summary, every pair summary
    /// that referenced `from` (each affected [`UnorderedPair`] rebuilt around `to`), and
    /// the channel's orientation in the fingerprint. It covers the case where a channel
    /// was recorded under the wrong key and its statistics are sound but mislabeled.
    ///
    /// When `to` already exists, the moved data and the existing data are *merged* with
    /// [`MeanVar::merge`], and the destination keeps its own model-version tag and
    /// orientation: the caller is asserting that `from`'s history belongs to the channel
    /// already living under `to`. A no-op `from == to` leaves the state unchanged. Unlike
    /// [`merge`](Self::merge), rekey does not run the tag gate; it is a deliberate rename
    /// and cannot fail.
    pub fn rekey(&mut self, from: &str, to: String) {
        if from == to.as_str() {
            return;
        }

        // Channel: move, merging into the destination if it already exists.
        if let Some(moved) = self.channels.remove(from) {
            match self.channels.get_mut(&to) {
                Some(existing) => {
                    existing.separation.merge_in(&moved.separation);
                    existing.reference.merge_in(&moved.reference);
                }
                None => {
                    self.channels.insert(to.clone(), moved);
                }
            }
        }

        // Pairs: rebuild every pair that mentioned `from`, merging on collision.
        let affected: Vec<UnorderedPair<String>> = self
            .pairs
            .keys()
            .filter(|p| p.first().as_str() == from || p.second().as_str() == from)
            .cloned()
            .collect();
        for old in affected {
            // `old` was just collected from this map's own keys, so the remove succeeds;
            // the `if let` keeps the public path free of a reachable panic regardless.
            if let Some(summary) = self.pairs.remove(&old) {
                let (a, b) = old.into_inner();
                let new_a = if a == from { to.clone() } else { a };
                let new_b = if b == from { to.clone() } else { b };
                let rebuilt = UnorderedPair::new(new_a, new_b);
                match self.pairs.get_mut(&rebuilt) {
                    Some(existing) => existing.redundancy.merge_in(&summary.redundancy),
                    None => {
                        self.pairs.insert(rebuilt, summary);
                    }
                }
            }
        }

        // Fingerprint orientation: move it, keeping the destination's if one is set.
        if let Some(dir) = self.fingerprint.directions.remove(from) {
            self.fingerprint.directions.entry(to).or_insert(dir);
        }
    }

    /// Scales the confidence of every persisted summary down by `factor`.
    ///
    /// Applies [`MeanVar::decay`] to each channel's separation and reference baselines and
    /// to each pair's redundancy, shrinking their effective counts while leaving their
    /// means and variances unchanged. `factor` is clamped to `[0, 1]` by
    /// [`MeanVar::decay`].
    ///
    /// Decay is the one operation that breaks the exactness of [`merge`](Self::merge): a
    /// decayed state carries a different effective count than an undecayed one, so decaying
    /// then merging no longer gives the same result as merging then decaying. It is gated
    /// behind [`DecayConfig`](crate::config::DecayConfig) and off by default.
    pub fn decay(&mut self, factor: f64) {
        for summary in self.channels.values_mut() {
            summary.separation.decay(factor);
            summary.reference.decay(factor);
        }
        for summary in self.pairs.values_mut() {
            summary.redundancy.decay(factor);
            summary.refreshes *= decay_factor(factor);
        }
    }
}

/// The clamped decay multiplier: `factor` held to `[0, 1]`, a non-finite factor treated
/// as `0`. Mirrors the clamping inside [`MeanVar::decay`] for the plain-`f64` refresh
/// count, so the two decay in lockstep.
pub(crate) fn decay_factor(factor: f64) -> f64 {
    if factor.is_finite() {
        factor.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The standardized distance between two baselines: `|Δmean| / pooled_std` with
/// `pooled_std = sqrt((var_a + var_b) / 2)`, capped at the finite divergence ceiling
/// `1e6`.
///
/// Zero pooled spread with equal means is `0.0`; zero spread with differing means is
/// the cap (an infinite z-distance, clamped). The result is always finite and
/// non-negative.
fn standardized_distance(a: &MeanVar, b: &MeanVar) -> f64 {
    let pooled_std = ((a.variance() + b.variance()) / 2.0).sqrt();
    let mean_diff = (a.mean() - b.mean()).abs();
    if pooled_std.is_finite() && pooled_std > 0.0 {
        (mean_diff / pooled_std).min(DIVERGENCE_CAP)
    } else if mean_diff == 0.0 {
        0.0
    } else {
        DIVERGENCE_CAP
    }
}

/// `serde` adapter that serializes the pair map as a sequence of `[pair, summary]`
/// entries rather than a JSON object.
///
/// `UnorderedPair<String>` derives a two-element tuple `Serialize`, which JSON
/// cannot use as an object key (keys must be strings). Writing the map as an array
/// keeps `RuffleState`'s field type frozen, round-trips exactly, and preserves the
/// canonical `BTreeMap` ordering that gives content-addressing. The companion
/// `channels` map needs no adapter: its join-handle key is a `String` and so
/// serializes straight to a string key.
mod pairs_as_seq {
    use super::{PairSummary, UnorderedPair};
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::collections::BTreeMap;

    pub(super) fn serialize<S>(
        pairs: &BTreeMap<UnorderedPair<String>, PairSummary>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(pairs.len()))?;
        for entry in pairs {
            seq.serialize_element(&entry)?;
        }
        seq.end()
    }

    pub(super) fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<BTreeMap<UnorderedPair<String>, PairSummary>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries: Vec<(UnorderedPair<String>, PairSummary)> = Vec::deserialize(deserializer)?;
        Ok(entries.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::BaselineMode;
    use approx::assert_abs_diff_eq;

    fn key(s: &str) -> String {
        s.to_string()
    }

    /// A channel summary tagged `t`, its separation baseline built by pushing `sep`.
    fn chan(t: &str, sep: &[f64]) -> ChannelSummary {
        let mut c = ChannelSummary::new(t.to_string());
        for &x in sep {
            c.separation.push(x);
        }
        c
    }

    fn fp(dirs: &[(&str, Direction)]) -> StatFingerprint {
        let mut m = BTreeMap::new();
        for (k, d) in dirs {
            m.insert(key(k), *d);
        }
        StatFingerprint::new(BaselineMode::ZScore, m)
    }

    /// A state carrying a fingerprint and a list of `(key, summary)` channels.
    fn state(fingerprint: StatFingerprint, chans: &[(&str, ChannelSummary)]) -> RuffleState {
        let mut s = RuffleState::new(fingerprint);
        for (k, c) in chans {
            s.channels.insert(key(k), c.clone());
        }
        s
    }

    #[test]
    fn merge_policy_default_is_strict() {
        assert_eq!(MergePolicy::default(), MergePolicy::Strict);
    }

    // --- The one merge: associative, commutative, and update = prior = reconcile ---

    #[test]
    fn merge_is_associative_and_commutative() {
        let dirs = &[
            ("x", Direction::HigherIsBetter),
            ("y", Direction::HigherIsBetter),
            ("z", Direction::LowerIsBetter),
        ];
        let a = state(
            fp(dirs),
            &[
                ("x", chan("mx", &[1.0, 2.0, 3.0])),
                ("y", chan("my", &[5.0, 6.0])),
            ],
        );
        let b = state(
            fp(dirs),
            &[("x", chan("mx", &[10.0, 11.0])), ("z", chan("mz", &[0.5]))],
        );
        let c = state(
            fp(dirs),
            &[
                ("x", chan("mx", &[100.0])),
                ("y", chan("my", &[7.0, 8.0, 9.0])),
            ],
        );

        let (m1, _) = RuffleState::merge(&[&a, &b, &c], MergePolicy::Strict).unwrap();
        let (m2, _) = RuffleState::merge(&[&c, &b, &a], MergePolicy::Strict).unwrap();
        let (m3, _) = RuffleState::merge(&[&b, &a, &c], MergePolicy::Strict).unwrap();

        // Union of all keys, in canonical order, regardless of fold order.
        let keys: Vec<&String> = m1.channels.keys().collect();
        assert_eq!(keys, vec![&key("x"), &key("y"), &key("z")]);

        // Means and variances are identical across orders up to f64 rounding (the §8
        // "update order never matters" claim).
        for k in m1.channels.keys() {
            let s1 = &m1.channels[k].separation;
            let s2 = &m2.channels[k].separation;
            let s3 = &m3.channels[k].separation;
            assert_abs_diff_eq!(s1.mean(), s2.mean(), epsilon = 1e-9);
            assert_abs_diff_eq!(s1.mean(), s3.mean(), epsilon = 1e-9);
            assert_abs_diff_eq!(s1.variance(), s2.variance(), epsilon = 1e-9);
            assert_abs_diff_eq!(s1.variance(), s3.variance(), epsilon = 1e-9);
            assert_abs_diff_eq!(s1.count(), s2.count(), epsilon = 1e-9);
        }
        // x saw 3 + 2 + 1 observations across the parts.
        assert_abs_diff_eq!(
            m1.channels[&key("x")].separation.count(),
            6.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn streaming_update_equals_prior_equals_reconcile() {
        let dirs = &[("x", Direction::HigherIsBetter)];

        // An operator prior: a hand-written summary with a pseudo-count.
        let mut prior_chan = ChannelSummary::new("m".to_string());
        prior_chan.separation = MeanVar::from_prior(0.3, 0.04, 4.0);
        let mut prior = state(fp(dirs), &[]);
        prior.channels.insert(key("x"), prior_chan);

        // Three queries, each a count-1 summary of channel x.
        let q = |v: f64| state(fp(dirs), &[("x", chan("m", &[v]))]);
        let (q1, q2, q3) = (q(0.5), q(0.9), q(0.1));

        // Reconcile in two different orders.
        let (fwd, _) = RuffleState::merge(&[&prior, &q1, &q2, &q3], MergePolicy::Strict).unwrap();
        let (perm, _) = RuffleState::merge(&[&q3, &q1, &prior, &q2], MergePolicy::Strict).unwrap();

        // Reference: fold the underlying summaries directly (a prior then streaming
        // updates), which is the same operation.
        let mut reference = MeanVar::from_prior(0.3, 0.04, 4.0);
        for v in [0.5, 0.9, 0.1] {
            reference.push(v);
        }

        for merged in [&fwd, &perm] {
            let sep = &merged.channels[&key("x")].separation;
            assert_abs_diff_eq!(sep.mean(), reference.mean(), epsilon = 1e-9);
            assert_abs_diff_eq!(sep.variance(), reference.variance(), epsilon = 1e-9);
            assert_abs_diff_eq!(sep.count(), 7.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn single_part_merge_returns_it() {
        let a = state(
            fp(&[("x", Direction::HigherIsBetter)]),
            &[("x", chan("m", &[1.0, 2.0, 3.0]))],
        );
        let (m, d) = RuffleState::merge(&[&a], MergePolicy::Strict).unwrap();
        assert_eq!(m, a);
        assert!(d.per_channel.is_empty());
        assert_eq!(d.max, 0.0);
    }

    // --- The union ---

    #[test]
    fn merge_unions_channels_and_directions() {
        let a = state(
            fp(&[
                ("x", Direction::HigherIsBetter),
                ("y", Direction::HigherIsBetter),
            ]),
            &[("x", chan("m", &[1.0, 2.0])), ("y", chan("m", &[3.0]))],
        );
        let b = state(
            fp(&[
                ("x", Direction::HigherIsBetter),
                ("z", Direction::LowerIsBetter),
            ]),
            &[("x", chan("m", &[10.0])), ("z", chan("m", &[4.0]))],
        );

        let (m, _) = RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap();

        let keys: Vec<&String> = m.channels.keys().collect();
        assert_eq!(keys, vec![&key("x"), &key("y"), &key("z")]);
        // x is the merge of both parts; y and z carry through untouched.
        assert_abs_diff_eq!(
            m.channels[&key("x")].separation.count(),
            3.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            m.channels[&key("y")].separation.mean(),
            3.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            m.channels[&key("z")].separation.mean(),
            4.0,
            epsilon = 1e-12
        );
        // Directions union, mixed orientations preserved.
        let dir_keys: Vec<&String> = m.fingerprint.directions.keys().collect();
        assert_eq!(dir_keys, vec![&key("x"), &key("y"), &key("z")]);
        assert_eq!(
            m.fingerprint.directions[&key("z")],
            Direction::LowerIsBetter
        );
    }

    #[test]
    fn merge_pairs_union_and_merge() {
        let mut a = state(fp(&[]), &[]);
        let mut pa = PairSummary::new();
        pa.redundancy.push(0.2);
        pa.redundancy.push(0.4);
        a.pairs.insert(UnorderedPair::new(key("x"), key("y")), pa);

        let mut b = state(fp(&[]), &[]);
        let mut pb = PairSummary::new();
        pb.redundancy.push(0.6);
        b.pairs.insert(UnorderedPair::new(key("y"), key("x")), pb); // same pair, other order

        let (m, _) = RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap();
        assert_eq!(m.pairs.len(), 1);
        let red = &m.pairs[&UnorderedPair::new(key("x"), key("y"))].redundancy;
        assert_abs_diff_eq!(red.count(), 3.0, epsilon = 1e-12);
    }

    // --- The gates: every incompatibility refuses ---

    #[test]
    fn empty_parts_refuses() {
        let parts: [&RuffleState; 0] = [];
        assert_eq!(
            RuffleState::merge(&parts, MergePolicy::Strict).unwrap_err(),
            Mismatch::Empty
        );
    }

    #[test]
    fn format_version_mismatch_refuses() {
        let a = state(fp(&[]), &[]);
        let mut b = state(fp(&[]), &[]);
        b.format_version = 99;
        assert_eq!(
            RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap_err(),
            Mismatch::FormatVersion {
                left: RuffleState::FORMAT_VERSION,
                right: 99
            }
        );
    }

    #[test]
    fn stale_build_version_refuses_even_when_parts_agree() {
        // Two states at the SAME stale format version parse fine and agree with each
        // other; the merge must still refuse, because "parses" is not "compatible with
        // this build" (§8). Before this gate, two v99 states merged cleanly under any
        // build.
        let mut a = state(fp(&[]), &[]);
        let mut b = state(fp(&[]), &[]);
        a.format_version = 99;
        b.format_version = 99;
        assert_eq!(
            RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap_err(),
            Mismatch::FormatVersion {
                left: RuffleState::FORMAT_VERSION,
                right: 99
            }
        );

        // Same for a stale statistic version under the current format version.
        let mut c = state(fp(&[]), &[]);
        let mut d = state(fp(&[]), &[]);
        c.fingerprint.stat_version = 1;
        d.fingerprint.stat_version = 1;
        assert_eq!(
            RuffleState::merge(&[&c, &d], MergePolicy::Strict).unwrap_err(),
            Mismatch::Fingerprint
        );
    }

    #[test]
    fn fingerprint_stat_version_mismatch_refuses() {
        let a = state(fp(&[]), &[]);
        let mut b = state(fp(&[]), &[]);
        b.fingerprint.stat_version = 99;
        assert_eq!(
            RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap_err(),
            Mismatch::Fingerprint
        );
    }

    #[test]
    fn per_channel_tag_mismatch_refuses() {
        let a = state(
            fp(&[("x", Direction::HigherIsBetter)]),
            &[("x", chan("model-1", &[1.0]))],
        );
        let b = state(
            fp(&[("x", Direction::HigherIsBetter)]),
            &[("x", chan("model-2", &[2.0]))],
        );
        assert_eq!(
            RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap_err(),
            Mismatch::Tag {
                channel: "x".to_string(),
                left: "model-1".to_string(),
                right: "model-2".to_string(),
            }
        );
    }

    #[test]
    fn direction_conflict_on_shared_channel_refuses() {
        // Same key, opposite orientation: a fingerprint-level incompatibility (§7).
        let a = state(fp(&[("x", Direction::HigherIsBetter)]), &[]);
        let b = state(fp(&[("x", Direction::LowerIsBetter)]), &[]);
        assert_eq!(
            RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap_err(),
            Mismatch::DirectionConflict {
                channel: "x".to_string()
            }
        );
    }

    #[test]
    fn differing_directions_on_different_channels_union_fine() {
        let a = state(fp(&[("x", Direction::HigherIsBetter)]), &[]);
        let b = state(fp(&[("y", Direction::LowerIsBetter)]), &[]);
        let (m, _) = RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap();
        assert_eq!(
            m.fingerprint.directions[&key("x")],
            Direction::HigherIsBetter
        );
        assert_eq!(
            m.fingerprint.directions[&key("y")],
            Direction::LowerIsBetter
        );
    }

    // --- Divergence: advisory, never gates ---

    #[test]
    fn divergence_of_identical_states_is_zero() {
        let a = state(
            fp(&[("x", Direction::HigherIsBetter)]),
            &[("x", chan("m", &[1.0, 2.0, 3.0, 4.0]))],
        );
        let b = a.clone();
        let d = a.divergence(&b);
        assert_eq!(d.max, 0.0);
        assert_eq!(d.per_channel[&key("x")], 0.0);
    }

    #[test]
    fn divergence_flags_a_shifted_channel_only() {
        let dirs = &[
            ("x", Direction::HigherIsBetter),
            ("y", Direction::HigherIsBetter),
        ];
        let a = state(
            fp(dirs),
            &[
                ("x", chan("m", &[-1.0, 0.0, 1.0])),
                ("y", chan("m", &[5.0, 6.0, 7.0])),
            ],
        );
        let b = state(
            fp(dirs),
            &[
                ("x", chan("m", &[9.0, 10.0, 11.0])), // shifted +10, same spread
                ("y", chan("m", &[5.0, 6.0, 7.0])),   // identical
            ],
        );
        let d = a.divergence(&b);
        assert!(
            d.per_channel[&key("x")] > 5.0,
            "x = {}",
            d.per_channel[&key("x")]
        );
        assert_abs_diff_eq!(d.per_channel[&key("y")], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.max, d.per_channel[&key("x")], epsilon = 0.0);
    }

    #[test]
    fn divergence_only_covers_shared_channels() {
        let a = state(
            fp(&[("x", Direction::HigherIsBetter)]),
            &[("x", chan("m", &[1.0, 2.0]))],
        );
        let b = state(
            fp(&[("y", Direction::HigherIsBetter)]),
            &[("y", chan("m", &[3.0, 4.0]))],
        );
        let d = a.divergence(&b);
        assert!(d.per_channel.is_empty());
        assert_eq!(d.max, 0.0);
    }

    #[test]
    fn standardized_distance_pins_formula_and_guard() {
        // The advisory distance is `|Δmean| / sqrt((var_a + var_b) / 2)`, capped at the
        // finite ceiling. Hand-computed values pin every operator in the formula:
        // var_a = 2, var_b = 6 -> pooled_std = sqrt((2+6)/2) = sqrt(4) = 2; mean gap 9 ->
        // distance = 9/2 = 4.5. Any arithmetic mutation (+ -> * or -, / -> * or %) or a
        // guard comparison that flips the live branch lands on the cap or a different
        // ratio, never 4.5.
        let a = MeanVar::from_prior(1.0, 2.0, 4.0);
        let b = MeanVar::from_prior(10.0, 6.0, 4.0);
        assert_abs_diff_eq!(standardized_distance(&a, &b), 4.5, epsilon = 1e-12);

        // Zero pooled spread with EQUAL means is the live boundary of the guard
        // `pooled_std.is_finite() && pooled_std > 0.0`: it must take the else branch and
        // report 0.0, not divide by zero. The `&&` -> `||` and `> 0.0` -> `>= 0.0`
        // mutants both enter the ratio branch, compute `0/0 = NaN`, and `min`-clamp it to
        // the cap -- distinguishable from 0.0 only here (with DIFFERING means the cap is
        // the correct answer, so that case cannot tell them apart).
        let z0 = MeanVar::from_prior(5.0, 0.0, 3.0);
        let z1 = MeanVar::from_prior(5.0, 0.0, 3.0);
        assert_eq!(standardized_distance(&z0, &z1), 0.0);
    }

    // The running-maximum updates `if d > *slot` (merge) and `if d > max` (divergence)
    // are EQUIVALENT mutants under `>` -> `>=`. Each accumulates the maximum of a set of
    // non-negative distances starting from 0; reassigning the slot when `d` equals the
    // current maximum stores the identical value, so the final maximum is the same for
    // `>` and `>=`. No input distinguishes them.

    #[test]
    fn divergence_catches_a_reference_shift_the_separation_misses() {
        // The silent-model-swap signature (§8): the separation baseline is scale- and
        // shift-invariant, so a swap that rescales scores can leave it untouched while
        // the native-units good-score reference jumps. The divergence must read the
        // larger of the two per-channel distances, so this swap is visible.
        let mk = |ref_mean: f64| {
            let mut c = chan("m", &[1.0, 2.0, 3.0]); // identical separation baselines
            c.reference = MeanVar::from_prior(ref_mean, 0.01, 6.0);
            state(fp(&[("x", Direction::HigherIsBetter)]), &[("x", c)])
        };
        let a = mk(0.30);
        let b = mk(0.90); // reference shifted 6 pooled-sigma; separation identical
        let d = a.divergence(&b);
        assert!(
            d.per_channel[&key("x")] > 5.0,
            "the reference shift must dominate: {}",
            d.per_channel[&key("x")]
        );
        // Identical references and separations still read zero.
        assert_eq!(mk(0.30).divergence(&mk(0.30)).max, 0.0);
    }

    #[test]
    fn merge_sums_pair_refreshes_and_decay_scales_them() {
        // Merging two states pools their between-refresh evidence: refresh counts add.
        let pair = UnorderedPair::new(key("x"), key("y"));
        let mut a = state(fp(&[]), &[]);
        let mut pa = PairSummary::new();
        pa.redundancy.push(0.2);
        pa.refreshes = 1.0;
        a.pairs.insert(pair.clone(), pa);

        let mut b = state(fp(&[]), &[]);
        let mut pb = PairSummary::new();
        pb.redundancy.push(0.4);
        pb.refreshes = 2.0;
        b.pairs.insert(pair.clone(), pb);

        let (m, _) = RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap();
        assert_abs_diff_eq!(m.pairs[&pair].refreshes, 3.0, epsilon = 1e-12);

        // Decay scales the refresh count alongside the redundancy count, so the two
        // gates age in lockstep.
        let mut decayed = m;
        decayed.decay(0.5);
        assert_abs_diff_eq!(decayed.pairs[&pair].refreshes, 1.5, epsilon = 1e-12);
    }

    #[test]
    fn divergence_caps_when_spread_collapses() {
        // Both baselines degenerate (zero variance) but with different means: an
        // infinite z-distance, reported at the cap.
        let a = state(
            fp(&[("x", Direction::HigherIsBetter)]),
            &[("x", chan("m", &[5.0, 5.0, 5.0]))],
        );
        let b = state(
            fp(&[("x", Direction::HigherIsBetter)]),
            &[("x", chan("m", &[8.0, 8.0, 8.0]))],
        );
        let d = a.divergence(&b);
        assert_abs_diff_eq!(d.per_channel[&key("x")], DIVERGENCE_CAP, epsilon = 0.0);
    }

    #[test]
    fn merge_reports_max_pairwise_divergence() {
        let mk = |vals: &[f64]| {
            state(
                fp(&[("x", Direction::HigherIsBetter)]),
                &[("x", chan("m", vals))],
            )
        };
        let a = mk(&[-1.0, 0.0, 1.0]);
        let b = mk(&[-1.0, 0.0, 1.0]); // identical to a
        let c = mk(&[9.0, 10.0, 11.0]); // shifted

        let (_, d) = RuffleState::merge(&[&a, &b, &c], MergePolicy::Strict).unwrap();
        // The a–b pair is 0, but the a–c / b–c pairs are large; the max wins.
        assert!(d.per_channel[&key("x")] > 5.0);
        assert_abs_diff_eq!(d.max, d.per_channel[&key("x")], epsilon = 0.0);
        assert_eq!(a.divergence(&b).per_channel[&key("x")], 0.0);
    }

    // --- rekey: the safe rename ---

    #[test]
    fn rekey_renames_across_channels_pairs_and_fingerprint() {
        let mut s = state(
            fp(&[
                ("old", Direction::LowerIsBetter),
                ("p2", Direction::HigherIsBetter),
            ]),
            &[
                ("old", chan("m", &[1.0, 2.0, 3.0])),
                ("p2", chan("m", &[4.0])),
            ],
        );
        let mut pair = PairSummary::new();
        pair.redundancy.push(0.5);
        s.pairs
            .insert(UnorderedPair::new(key("old"), key("p2")), pair);
        let old_mean = s.channels[&key("old")].separation.mean();

        s.rekey(&key("old"), key("new"));

        assert!(!s.channels.contains_key(&key("old")));
        assert_abs_diff_eq!(
            s.channels[&key("new")].separation.mean(),
            old_mean,
            epsilon = 1e-12
        );
        // The pair is rebuilt around the new key.
        assert!(
            s.pairs
                .contains_key(&UnorderedPair::new(key("new"), key("p2")))
        );
        assert!(
            !s.pairs
                .contains_key(&UnorderedPair::new(key("old"), key("p2")))
        );
        // The orientation moves with the key.
        assert_eq!(
            s.fingerprint.directions.get(&key("new")),
            Some(&Direction::LowerIsBetter)
        );
        assert!(!s.fingerprint.directions.contains_key(&key("old")));
    }

    #[test]
    fn rekey_into_existing_key_merges_keeping_destination_tag() {
        let mut s = state(
            fp(&[
                ("old", Direction::HigherIsBetter),
                ("new", Direction::HigherIsBetter),
            ]),
            &[
                ("old", chan("m", &[1.0, 2.0, 3.0])),
                ("new", chan("keep", &[10.0, 11.0])),
            ],
        );
        s.rekey(&key("old"), key("new"));
        assert!(!s.channels.contains_key(&key("old")));
        let merged = &s.channels[&key("new")];
        assert_abs_diff_eq!(merged.separation.count(), 5.0, epsilon = 1e-12);
        // Collision keeps the destination's identity.
        assert_eq!(merged.tag.as_str(), "keep");
    }

    #[test]
    fn rekey_merges_colliding_pairs() {
        let mut s = state(
            fp(&[]),
            &[
                ("old", chan("m", &[1.0])),
                ("new", chan("m", &[2.0])),
                ("x", chan("m", &[3.0])),
            ],
        );
        let mut p_old = PairSummary::new();
        p_old.redundancy.push(0.2);
        p_old.redundancy.push(0.4);
        let mut p_new = PairSummary::new();
        p_new.redundancy.push(0.6);
        s.pairs
            .insert(UnorderedPair::new(key("old"), key("x")), p_old);
        s.pairs
            .insert(UnorderedPair::new(key("new"), key("x")), p_new);

        s.rekey(&key("old"), key("new"));

        assert_eq!(s.pairs.len(), 1);
        let merged = &s.pairs[&UnorderedPair::new(key("new"), key("x"))].redundancy;
        assert_abs_diff_eq!(merged.count(), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn rekey_roundtrips_without_collision() {
        let mut s = state(
            fp(&[("a", Direction::HigherIsBetter)]),
            &[("a", chan("m", &[1.0, 2.0, 3.0]))],
        );
        let before = s.channels[&key("a")].separation.mean();
        s.rekey(&key("a"), key("b"));
        s.rekey(&key("b"), key("a"));
        assert!(s.channels.contains_key(&key("a")));
        assert_abs_diff_eq!(
            s.channels[&key("a")].separation.mean(),
            before,
            epsilon = 1e-12
        );
    }

    #[test]
    fn rekey_is_a_noop_when_from_equals_to() {
        let mut s = state(
            fp(&[("a", Direction::HigherIsBetter)]),
            &[("a", chan("m", &[1.0, 2.0]))],
        );
        let before = s.clone();
        s.rekey(&key("a"), key("a"));
        assert_eq!(s, before);
    }

    // --- decay ---

    #[test]
    fn decay_halves_counts_and_preserves_means_and_variances() {
        let mut x = chan("m", &[1.0, 2.0, 3.0, 4.0]);
        x.reference = MeanVar::from_prior(0.5, 0.01, 6.0);
        let mut s = state(fp(&[("x", Direction::HigherIsBetter)]), &[]);
        s.channels.insert(key("x"), x);
        let mut pair = PairSummary::new();
        pair.redundancy.push(0.3);
        pair.redundancy.push(0.5);
        s.pairs.insert(UnorderedPair::new(key("x"), key("x")), pair);

        let sep0 = s.channels[&key("x")].separation;
        let ref0 = s.channels[&key("x")].reference;
        let red0 = s.pairs[&UnorderedPair::new(key("x"), key("x"))].redundancy;

        s.decay(0.5);

        let sep1 = &s.channels[&key("x")].separation;
        assert_abs_diff_eq!(sep1.count(), sep0.count() * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(sep1.mean(), sep0.mean(), epsilon = 1e-12);
        assert_abs_diff_eq!(sep1.variance(), sep0.variance(), epsilon = 1e-12);

        let ref1 = &s.channels[&key("x")].reference;
        assert_abs_diff_eq!(ref1.count(), ref0.count() * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(ref1.mean(), ref0.mean(), epsilon = 1e-12);
        assert_abs_diff_eq!(ref1.variance(), ref0.variance(), epsilon = 1e-12);

        let red1 = &s.pairs[&UnorderedPair::new(key("x"), key("x"))].redundancy;
        assert_abs_diff_eq!(red1.count(), red0.count() * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(red1.mean(), red0.mean(), epsilon = 1e-12);
    }

    // --- Serialization: canonical, content-addressing ---

    #[test]
    fn serde_json_round_trips_exactly() {
        let mut y = chan("model-y", &[4.0, 5.0]);
        y.reference = MeanVar::from_prior(0.4, 0.02, 3.0);
        let mut s = state(
            fp(&[
                ("x", Direction::HigherIsBetter),
                ("y", Direction::LowerIsBetter),
            ]),
            &[("x", chan("model-x", &[1.0, 2.0, 3.0]))],
        );
        s.channels.insert(key("y"), y);
        let mut pair = PairSummary::new();
        pair.redundancy.push(0.25);
        s.pairs.insert(UnorderedPair::new(key("x"), key("y")), pair);

        let json = serde_json::to_string(&s).unwrap();
        let back: RuffleState = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn identical_contents_serialize_byte_identically() {
        // Built with channels inserted in opposite orders; BTreeMap canonicalizes, so
        // the JSON is byte-identical (content-addressing, §8).
        let mut s1 = RuffleState::new(fp(&[
            ("x", Direction::HigherIsBetter),
            ("y", Direction::HigherIsBetter),
        ]));
        s1.channels.insert(key("y"), chan("m", &[3.0, 4.0]));
        s1.channels.insert(key("x"), chan("m", &[1.0, 2.0]));

        let mut s2 = RuffleState::new(fp(&[
            ("x", Direction::HigherIsBetter),
            ("y", Direction::HigherIsBetter),
        ]));
        s2.channels.insert(key("x"), chan("m", &[1.0, 2.0]));
        s2.channels.insert(key("y"), chan("m", &[3.0, 4.0]));

        assert_eq!(
            serde_json::to_string(&s1).unwrap(),
            serde_json::to_string(&s2).unwrap()
        );
    }
}
