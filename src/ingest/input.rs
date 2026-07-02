//! One query's input for one channel, in canonical higher-is-better units.
//!
//! A channel is either scored or ranks-only, a stable property of how the channel is
//! wired rather than something inferred per query. Ingest orients to higher-is-better
//! and drops non-finite values, so everything downstream sees canonical scores.

use crate::config::ChannelConfig;
use crate::score::{Score, orient, sanitize};

/// A channel's surfaced items for one query.
///
/// `Scored` holds scores that have been oriented to higher-is-better and had non-finite
/// values removed. An item the channel did not surface is left out rather than charged
/// a worst-rank penalty. `Ranks` is a ranks-only channel that produces no scores: it
/// cannot contribute a discrimination estimate, so it is weighted at the default and
/// flagged.
///
/// Constructing a variant directly (for example `Items::Scored(..)`) stores the values
/// verbatim and skips the orientation and non-finite filtering that
/// [`ChannelInput::scored`] performs. A caller building one by hand must supply scores
/// that are already finite and oriented higher-is-better.
///
/// Precondition: each channel lists each item at most once. A repeated id within one
/// channel's list is counted twice by the fusion, so the ids in a single `Items` must be
/// distinct.
#[derive(Clone, Debug, PartialEq)]
pub enum Items<Id> {
    /// Surfaced items paired with oriented, sanitized scores. Order is as supplied.
    Scored(Vec<(Id, f64)>),
    /// A ranks-only channel: items in rank order, best first, no scores.
    Ranks(Vec<Id>),
}

impl<Id> Items<Id> {
    /// The number of items carried.
    pub fn len(&self) -> usize {
        match self {
            Items::Scored(v) => v.len(),
            Items::Ranks(v) => v.len(),
        }
    }

    /// Whether the channel surfaced no items for this query.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this is a ranks-only channel, which carries no scores and so no
    /// discrimination estimate.
    pub fn is_ranks_only(&self) -> bool {
        matches!(self, Items::Ranks(_))
    }
}

/// One channel's input for one query: the channel's `key` plus its surfaced items.
///
/// Building this with a struct literal skips the orientation and non-finite filtering
/// that [`ChannelInput::scored`] and [`ChannelInput::ranked`] perform; `items` filled by
/// hand must already be finite and oriented higher-is-better. A channel is identified
/// per query only by its `key`; its model-version tag lives in the channel's
/// configuration and accumulated state, not in per-query input.
///
/// Precondition: each channel lists each item at most once. A repeated id within one
/// input's `items` is counted twice by the fusion, so the ids in one input must be
/// distinct.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelInput<Id> {
    /// The channel this input belongs to, named by its `key`.
    pub key: String,
    /// The surfaced items, already oriented and sanitized.
    pub items: Items<Id>,
}

impl<Id> ChannelInput<Id> {
    /// Builds a scored input: reads each item's score, orients it to higher-is-better by
    /// the channel's declared direction, and drops the item if the result is non-finite.
    ///
    /// The channel's `key` and direction are taken from `cfg`; the direction is the
    /// channel's configured one and cannot be overridden per call. A non-finite score
    /// carries no usable magnitude and would corrupt any baseline it reached.
    pub fn scored<S: Score>(cfg: &ChannelConfig, items: Vec<(Id, S)>) -> Self {
        let items = items
            .into_iter()
            .filter_map(|(id, s)| sanitize(orient(cfg.direction, s.value())).map(|x| (id, x)))
            .collect();
        Self {
            key: cfg.id.key.clone(),
            items: Items::Scored(items),
        }
    }

    /// Builds a ranks-only input for a channel that produces no scores.
    ///
    /// The channel's `key` is taken from `cfg`. The order is used as given, best first;
    /// there is nothing to orient or filter, since a rank carries no magnitude.
    pub fn ranked(cfg: &ChannelConfig, ids: Vec<Id>) -> Self {
        Self {
            key: cfg.id.key.clone(),
            items: Items::Ranks(ids),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::ChannelId;
    use crate::score::Direction;

    struct Raw(f64);
    impl Score for Raw {
        fn value(&self) -> f64 {
            self.0
        }
    }

    fn cfg(dir: Direction) -> ChannelConfig {
        ChannelConfig::new(ChannelId::new("ch", "tag-1"), dir, None)
    }

    fn scored_or_panic(obs: &ChannelInput<u32>) -> &Vec<(u32, f64)> {
        match &obs.items {
            Items::Scored(v) => v,
            Items::Ranks(_) => panic!("expected Scored"),
        }
    }

    #[test]
    fn higher_is_better_passes_scores_through() {
        let obs = ChannelInput::scored(
            &cfg(Direction::HigherIsBetter),
            vec![(1u32, Raw(0.9)), (2, Raw(0.1)), (3, Raw(-0.5))],
        );
        assert_eq!(
            scored_or_panic(&obs),
            &vec![(1u32, 0.9), (2, 0.1), (3, -0.5)]
        );
        assert_eq!(obs.key, "ch");
    }

    #[test]
    fn lower_is_better_negates_scores() {
        let obs = ChannelInput::scored(
            &cfg(Direction::LowerIsBetter),
            vec![(1u32, Raw(0.9)), (2, Raw(0.1)), (3, Raw(-0.5))],
        );
        assert_eq!(
            scored_or_panic(&obs),
            &vec![(1u32, -0.9), (2, -0.1), (3, 0.5)]
        );
    }

    #[test]
    fn non_finite_items_are_dropped() {
        let obs = ChannelInput::scored(
            &cfg(Direction::HigherIsBetter),
            vec![
                (1u32, Raw(0.5)),
                (2, Raw(f64::NAN)),
                (3, Raw(f64::INFINITY)),
                (4, Raw(f64::NEG_INFINITY)),
                (5, Raw(0.25)),
            ],
        );
        assert_eq!(scored_or_panic(&obs), &vec![(1u32, 0.5), (5, 0.25)]);
    }

    #[test]
    fn ranks_only_preserved() {
        let obs = ChannelInput::ranked(&cfg(Direction::HigherIsBetter), vec![10u32, 20, 30]);
        assert!(obs.items.is_ranks_only());
        match &obs.items {
            Items::Ranks(v) => assert_eq!(v, &vec![10, 20, 30]),
            Items::Scored(_) => panic!("expected Ranks"),
        }
    }

    #[test]
    fn items_len_and_empty() {
        let scored: Items<u32> = Items::Scored(vec![(1, 0.1)]);
        assert_eq!(scored.len(), 1);
        assert!(!scored.is_empty());
        let empty: Items<u32> = Items::Ranks(vec![]);
        assert!(empty.is_empty());
        assert!(empty.is_ranks_only());
    }
}
