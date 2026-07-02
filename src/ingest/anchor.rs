//! The full-scored anchor for coupling estimation.
//!
//! The anchor is a set of representative queries, each scored by every channel over a
//! common, unselected candidate set rather than any channel's top-`k`. Because it is
//! full-scored, an absent score is unambiguous: the facet does not apply to that item,
//! not that it ranked below a cutoff. The redundancy estimate is read here, away from
//! the live pool's selection bias.

use crate::config::ChannelConfig;
use crate::score::{Score, orient, sanitize};

/// A shared evaluation set in which every candidate is scored by every channel, used to
/// estimate how redundant the channels are with each other.
///
/// Because every candidate is scored by every channel, a `None` entry unambiguously means
/// the channel's facet does not apply to that item, rather than that the item was ranked
/// below a cutoff and dropped. The candidate set must be an unselected sample (a random
/// or whole-corpus draw) rather than any channel's top-`k` results; the type cannot
/// enforce this, so it is a caller precondition. Restricting the candidates to a top-`k`
/// pool biases the pairwise correlations and destroys the redundancy estimate.
#[derive(Debug, Clone)]
pub struct Anchor {
    /// The channels scored (by join-handle `key`), in a fixed order shared by every
    /// score row.
    pub(crate) channels: Vec<String>,
    /// `scores[channel][candidate]`: `Some(oriented score)` when the facet applies,
    /// `None` when it is absent for that item.
    pub(crate) scores: Vec<Vec<Option<f64>>>,
}

impl Anchor {
    /// Builds an anchor by scoring every `(candidate, channel)` pair.
    ///
    /// `score` is called once for each candidate and channel. `Some(s)` is oriented to
    /// higher-is-better by that channel's [`Direction`](crate::score::Direction); a finite
    /// result is stored as `Some`, a non-finite one as `None`. A `None` return is stored
    /// as-is and means the channel's facet does not apply to that candidate. Coverage is
    /// structural: because the closure runs for every pair, the anchor is always
    /// full-scored, and an absent score is never a hidden top-`k` cutoff.
    ///
    /// The candidate type `Id` is generic on this method alone: it is passed to `score` to
    /// address each candidate and is never stored, so the built [`Anchor`] is a plain score
    /// matrix.
    ///
    /// # Precondition
    ///
    /// `candidates` must be an unselected set: a random or whole-corpus draw rather
    /// than any channel's top-`k` pool. Restricting the candidates to a union of the channels'
    /// top-`k` results conditions on a selection effect (Berkson's paradox) that pushes the
    /// channels spuriously anti-correlated and destroys the redundancy estimate. Whether a
    /// candidate set is unselected cannot be checked from the ids alone, so keeping this
    /// contract is up to the caller.
    pub fn build<Id, S: Score>(
        candidates: &[Id],
        channels: &[&ChannelConfig],
        score: impl Fn(&Id, &str) -> Option<S>,
    ) -> Anchor {
        let channel_keys: Vec<String> = channels.iter().map(|c| c.id.key.clone()).collect();
        let scores: Vec<Vec<Option<f64>>> = channels
            .iter()
            .map(|cfg| {
                candidates
                    .iter()
                    .map(|id| {
                        // Orient by the channel's declared direction, then drop a
                        // non-finite value; `None` (facet absent) passes through.
                        score(id, cfg.id.key.as_str())
                            .and_then(|s| sanitize(orient(cfg.direction, s.value())))
                    })
                    .collect()
            })
            .collect();
        Anchor {
            channels: channel_keys,
            scores,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::ChannelId;
    use crate::score::Direction;

    /// A caller-side newtype: the only way a bare number becomes a [`Score`].
    struct Val(f64);
    impl Score for Val {
        fn value(&self) -> f64 {
            self.0
        }
    }

    fn channel(key: &str, dir: Direction) -> ChannelConfig {
        ChannelConfig::new(ChannelId::new(key, "tag"), dir, None)
    }

    #[test]
    fn build_covers_every_candidate_and_channel() {
        let cands: Vec<usize> = (0..5).collect();
        let a = channel("a", Direction::HigherIsBetter);
        let b = channel("b", Direction::HigherIsBetter);
        let anchor = Anchor::build(&cands, &[&a, &b], |id, _key| Some(Val(*id as f64)));
        // Two channels, each a full row over the five candidates.
        assert_eq!(anchor.channels.len(), 2);
        assert_eq!(anchor.scores.len(), 2);
        assert!(anchor.scores.iter().all(|row| row.len() == 5));
        // Every entry present (nothing truncated).
        assert!(anchor.scores.iter().flatten().all(|s| s.is_some()));
    }

    #[test]
    fn lower_is_better_channel_is_negated_at_ingest() {
        let cands = vec![0usize];
        let lo = channel("lo", Direction::LowerIsBetter);
        let anchor = Anchor::build(&cands, &[&lo], |_, _| Some(Val(3.0)));
        // A LowerIsBetter native score is oriented to higher-is-better by negation.
        assert_eq!(anchor.scores[0][0], Some(-3.0));
    }

    #[test]
    fn none_score_is_stored_as_absent_facet() {
        let cands: Vec<usize> = (0..4).collect();
        let a = channel("a", Direction::HigherIsBetter);
        // Facet absent on the odd candidates.
        let anchor = Anchor::build(&cands, &[&a], |id, _| {
            if id % 2 == 0 {
                Some(Val(*id as f64))
            } else {
                None
            }
        });
        assert_eq!(anchor.scores[0][0], Some(0.0));
        assert_eq!(anchor.scores[0][1], None);
        assert_eq!(anchor.scores[0][2], Some(2.0));
        assert_eq!(anchor.scores[0][3], None);
    }

    #[test]
    fn non_finite_score_is_dropped_to_none() {
        let cands = vec![0usize, 1];
        let a = channel("a", Direction::HigherIsBetter);
        let anchor = Anchor::build(&cands, &[&a], |id, _| {
            if *id == 0 {
                Some(Val(f64::NAN))
            } else {
                Some(Val(1.5))
            }
        });
        // A NaN that would corrupt a streaming correlation is sanitized to absent.
        assert_eq!(anchor.scores[0][0], None);
        assert_eq!(anchor.scores[0][1], Some(1.5));
    }
}
