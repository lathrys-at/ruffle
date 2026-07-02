//! Identifiers and the state fingerprint.
//!
//! A channel is named by a [`ChannelId`]: a stable join handle (the `key`) that keys
//! every persistent map, and a model-version `tag` that gates every merge. The
//! `tag` travels in the per-channel summary and is checked for equality on merge. A
//! [`StatFingerprint`] records whether two states were measuring the same thing the
//! same way.

use crate::score::Direction;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// A channel's identity: a stable join handle (`key`) plus a model-version `tag`.
///
/// The two fields serve different roles:
///
/// - `key` is the stable join handle. Every persistent map is keyed by it alone, so
///   accumulation across time and deployments lands on the right channel. It stays
///   fixed across model versions. A changed key mislabels statistics, recoverable by
///   rekeying or a cold start.
/// - `tag` is the model version (for example `"clip-vit-b32-rev1"`), changed whenever
///   the model behind the channel changes. Ruffle never interprets it; it only checks
///   it for equality on every merge. Two states that share a channel's `key` but
///   disagree on its `tag` are refused with
///   [`Mismatch::Tag`](crate::error::Mismatch::Tag), catching a model swapped in under
///   a kept key. An unnecessary tag change costs a cold start; a missed one corrupts
///   the baseline.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChannelId {
    /// The stable join handle. Every persistent map is keyed by this alone.
    pub key: String,
    /// The model-version tag. Ruffle never interprets it; it only requires that two
    /// states agree on it before their statistics for the same channel are merged, so a
    /// model swapped under a kept key is refused rather than silently blended.
    pub tag: String,
}

impl ChannelId {
    /// Builds a channel id from a join handle and a model-version tag.
    pub fn new(key: impl Into<String>, tag: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            tag: tag.into(),
        }
    }
}

impl fmt::Display for ChannelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.key, self.tag)
    }
}

/// An unordered pair, stored in canonical (sorted) order so `(a, b) == (b, a)`.
///
/// Used to key pairwise summaries by channel pair without regard to which member came
/// first. The constructor sorts its two members, so pairs with the same members in either
/// order compare, hash, and order identically.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UnorderedPair<T>(T, T);

impl<T: Ord> UnorderedPair<T> {
    /// Builds a pair, sorting the two members into canonical order.
    pub fn new(a: T, b: T) -> Self {
        if a <= b { Self(a, b) } else { Self(b, a) }
    }

    /// The smaller member (canonical first).
    pub fn first(&self) -> &T {
        &self.0
    }

    /// The larger member (canonical second).
    pub fn second(&self) -> &T {
        &self.1
    }

    /// Both members as a tuple, in canonical order.
    pub fn as_tuple(&self) -> (&T, &T) {
        (&self.0, &self.1)
    }

    /// Consumes the pair into its two members, in canonical order.
    pub fn into_inner(self) -> (T, T) {
        (self.0, self.1)
    }
}

/// How a channel's scores are standardized within the channel before comparison.
///
/// Only z-score standardization ships today; a mergeable quantile sketch is a planned
/// robustness upgrade.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BaselineMode {
    /// Standardizes each score against the channel's running mean and variance.
    #[default]
    ZScore,
}

/// A fingerprint answering whether two states were measuring the same thing the same way.
///
/// Two states built with different statistic definitions, orientations, or baseline
/// modes are numerically incompatible even when their serialization formats match.
/// Reconciliation refuses on a fingerprint mismatch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatFingerprint {
    /// Version of the discrimination/coupling statistic definitions. It increments
    /// when the meaning of a persisted statistic changes.
    pub stat_version: u32,
    /// Which within-channel standardization the state was built with.
    pub baseline_mode: BaselineMode,
    /// The per-channel orientation in force when the state was built, keyed by the
    /// channel's join handle. A direction change shows up here as a reason two states
    /// cannot be merged.
    pub directions: BTreeMap<String, Direction>,
}

impl StatFingerprint {
    /// The statistic-definition version this build of Ruffle writes.
    ///
    /// Version history:
    /// - `1`: Pearson anchor redundancy on raw oriented scores.
    /// - `2`: rank-based anchor redundancy (Spearman mapped to the Gaussian-copula
    ///   linear correlation), invariant to any monotone rescaling of a channel's
    ///   scores. Pair summaries accumulated under version `1` measure a different
    ///   statistic and do not merge with version-`2` state.
    pub const STAT_VERSION: u32 = 2;

    /// A fingerprint at the current statistic version with the given baseline mode
    /// and per-channel directions (keyed by join handle).
    pub fn new(baseline_mode: BaselineMode, directions: BTreeMap<String, Direction>) -> Self {
        Self {
            stat_version: Self::STAT_VERSION,
            baseline_mode,
            directions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_id_round_trips_and_displays() {
        let id = ChannelId::new("lexical", "bm25-v1");
        assert_eq!(id.key, "lexical");
        assert_eq!(id.tag, "bm25-v1");
        assert_eq!(id.to_string(), "lexical@bm25-v1");
        assert_eq!(
            id,
            ChannelId::new(String::from("lexical"), String::from("bm25-v1"))
        );
    }

    #[test]
    fn channel_id_orders_by_key_then_tag() {
        // Ord derives field order: key first, then tag. Two ids sharing a key order by tag.
        let a = ChannelId::new("lexical", "v1");
        let b = ChannelId::new("lexical", "v2");
        let c = ChannelId::new("recency", "v1");
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn unordered_pair_is_canonical() {
        let a = String::from("a");
        let b = String::from("b");
        let ab = UnorderedPair::new(a.clone(), b.clone());
        let ba = UnorderedPair::new(b, a);
        assert_eq!(ab, ba);
        assert_eq!(ab.first().as_str(), "a");
        assert_eq!(ab.second().as_str(), "b");
        // A pair keyed in one order is found by the same pair built in the other.
        let mut map = BTreeMap::new();
        map.insert(ab, 1u32);
        assert_eq!(map.get(&ba), Some(&1));
    }

    #[test]
    fn unordered_pair_same_member() {
        let p = UnorderedPair::new(3i32, 3i32);
        assert_eq!(p.first(), &3);
        assert_eq!(p.second(), &3);
    }

    #[test]
    fn unordered_pair_as_tuple_and_into_inner_are_canonical() {
        // Built out of order, both accessors report canonical (sorted) order.
        let p = UnorderedPair::new(5i32, 2i32);
        assert_eq!(p.as_tuple(), (&2, &5));
        // as_tuple borrows, so into_inner can still consume the same pair.
        assert_eq!(p.into_inner(), (2, 5));
    }

    #[test]
    fn fingerprint_carries_version() {
        let mut dirs = BTreeMap::new();
        dirs.insert(String::from("a"), Direction::HigherIsBetter);
        let fp = StatFingerprint::new(BaselineMode::ZScore, dirs);
        assert_eq!(fp.stat_version, StatFingerprint::STAT_VERSION);
        assert_eq!(fp.baseline_mode, BaselineMode::ZScore);
    }
}
