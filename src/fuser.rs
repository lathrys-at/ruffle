//! The top-level fuser that ties discrimination, coupling, and fusion together (§11).
//!
//! [`Fuser::fuse`] is the stateful entry point: orient, score, weight, fuse, then update
//! the persistent baselines.

use crate::config::{ChannelConfig, FuseConfig};
use crate::error::{ConfigError, Mismatch, ResumeError};
use crate::ingest::anchor::Anchor;
use crate::ingest::input::{ChannelInput, Items};
use crate::keys::{StatFingerprint, UnorderedPair};
use crate::state::{ChannelSummary, PairSummary, RuffleState, decay_factor};
use crate::summary::MeanVar;
use crate::weighting::NEUTRAL_WEIGHT;
use crate::weighting::coupling::{
    Diagnostics, PairBaseline, anchor_correlations, coupled_weights, diagnostics,
};
use crate::weighting::discrimination::{ChannelDiscrimination, discriminate, winsorize_separation};
use crate::weighting::fusion::weighted_rrf;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;

/// Why a channel was not weighted by its full discrimination score.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ChannelFlag {
    /// The channel supplied ranks only, with no scores to compute a discrimination
    /// statistic from, so it was carried at the neutral default weight.
    RanksOnlyDefaultWeighted,
    /// The score pool's bulk had no usable scale to measure the top's elevation against,
    /// so the separation read was floored rather than trusted.
    DegenerateSeparation,
    /// The channel had no usable good-score reference yet, so its absolute-goodness term
    /// could not be computed this query and it was weighted on separation alone.
    NoReference,
}

/// The outcome of fusing one query: the merged ranking plus the weights, flags, and
/// diagnostics behind it.
///
/// Marked `#[non_exhaustive]`: this is a result type read by callers, never built by
/// them, and it grows fields as the fusion surfaces more of its reasoning (the
/// `discrimination` map arrived that way). Construct rankings only through
/// [`Fuser::fuse`]/[`Fuser::fuse_stateless`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Fused<Id> {
    /// The fused ranking, best first, each item with its fused score.
    pub ranking: Vec<(Id, f64)>,
    /// The per-channel weights actually used, for inspection and debugging.
    pub weights: BTreeMap<String, f64>,
    /// Per-channel flags explaining any non-standard weighting; a channel absent from
    /// the map was weighted on its full discrimination score.
    pub flags: BTreeMap<String, ChannelFlag>,
    /// The per-channel discrimination reads behind the weights: the raw separation, the
    /// top-`m` reference read, and the combined `g`, per fused channel. Surfaced so
    /// "why did this channel get this weight" is answerable from the result alone,
    /// without recomputing through [`components`](crate::components).
    pub discrimination: BTreeMap<String, ChannelDiscrimination>,
    /// How much the discriminating channels agree on which items are relevant, as the
    /// Jaccard overlap of their top-ranked sets. Ranges from `0.0` to `1.0`.
    pub confidence: f64,
    /// One minus `confidence`, over the same discriminating channels. High when those
    /// channels each rank confidently but disagree on which items are relevant, so the
    /// choice of fusion matters most. Ranges from `0.0` to `1.0`.
    pub conflict: f64,
}

/// The entry point: fuses several retrieval channels' ranked outputs into one ranking.
///
/// A `Fuser` holds the channel registrations, the fusion configuration, and the
/// persistent baselines it accumulates across queries. Each [`fuse`](Self::fuse) call
/// weights the channels by how well each is discriminating on this query and how
/// redundant the channels are with each other, then combines them by weighted
/// reciprocal-rank fusion. Build one with [`new`](Self::new), or [`resume`](Self::resume)
/// from saved state; call [`fuse`](Self::fuse) per query; persist [`state`](Self::state).
#[derive(Debug, Clone)]
pub struct Fuser {
    /// Per-channel registrations, keyed by channel.
    configs: BTreeMap<String, ChannelConfig>,
    /// The persistent baseline state, updated on each stateful fuse.
    state: RuffleState,
    /// The fusion configuration.
    cfg: FuseConfig,
}

impl Fuser {
    /// Build a fresh fuser from channel registrations and a configuration, with empty
    /// starting baselines.
    ///
    /// The per-channel lookup is keyed by each config's join handle `id.key`, and the
    /// empty [`RuffleState`] is fingerprinted from `cfg.baseline_mode` and the
    /// `{id.key -> direction}` map of these configs, so the caller does not have to
    /// assemble a `BTreeMap` or a [`StatFingerprint`]/[`RuffleState`] by hand for the
    /// fresh case. To resume from a previously persisted state, use
    /// [`Fuser::resume`](Self::resume).
    ///
    /// # Errors
    ///
    /// Refuses an invalid configuration ([`FuseConfig::validate`]), a duplicate channel
    /// key (two registrations would write to the same baseline), or a declared
    /// [`GoodScore`](crate::score::GoodScore) that does not orient to a usable reference
    /// (which would otherwise silently cold-start the channel as if nothing had been
    /// declared).
    ///
    /// # Examples
    ///
    /// ```
    /// use ruffle::{ChannelConfig, ChannelId, Direction, FuseConfig, Fuser};
    ///
    /// let lexical = ChannelConfig::new(
    ///     ChannelId::new("lexical", "bm25-v1"),
    ///     Direction::HigherIsBetter,
    ///     None,
    /// );
    /// let dense = ChannelConfig::new(
    ///     ChannelId::new("dense", "clip-v1"),
    ///     Direction::HigherIsBetter,
    ///     None,
    /// );
    ///
    /// // A fresh fuser over two channels at the conservative default configuration.
    /// let fuser = Fuser::new(&[lexical, dense], FuseConfig::default())?;
    /// // Both channels are registered: the fresh state's fingerprint records their
    /// // orientations, and the persisted state is read through `state()`.
    /// assert_eq!(fuser.state().fingerprint().directions.len(), 2);
    /// # Ok::<(), ruffle::ConfigError>(())
    /// ```
    pub fn new(configs: &[ChannelConfig], cfg: FuseConfig) -> Result<Self, ConfigError> {
        validate_registrations(configs, &cfg)?;
        let mut directions = BTreeMap::new();
        for c in configs {
            directions.insert(c.id.key.clone(), c.direction);
        }
        let state = RuffleState::new(StatFingerprint::new(cfg.baseline_mode, directions));
        Ok(Self {
            configs: config_lookup(configs),
            state,
            cfg,
        })
    }

    /// Build a fuser from channel registrations, a previously persisted state, and a
    /// configuration, continuing to accumulate from that state.
    ///
    /// Builds the same per-channel lookup as [`Fuser::new`](Self::new), but uses the
    /// given `state` rather than an empty one, so accumulation continues from where the
    /// persisted state left off.
    ///
    /// Resume is the live boundary a real model change crosses (a swap happens across a
    /// restart), so it runs the same compatibility gate a state merge does before
    /// accepting the state. Without it, a model swapped in behind a dutifully bumped
    /// tag would silently keep accumulating into the old model's baselines, which is
    /// exactly the falsely-same corruption the tag exists to prevent (§8).
    ///
    /// # Errors
    ///
    /// Refuses everything [`Fuser::new`](Self::new) refuses, plus any incompatibility
    /// between the registrations and the state:
    ///
    /// - a state written at another format or statistic version, or under a different
    ///   baseline mode ([`Mismatch::FormatVersion`] / [`Mismatch::Fingerprint`]);
    /// - a channel whose configured direction contradicts the state fingerprint
    ///   ([`Mismatch::DirectionConflict`]);
    /// - a channel whose configured tag differs from the tag its accumulated statistics
    ///   were measured under ([`Mismatch::Tag`]): the signature of a model swap. Bumping
    ///   the tag was correct; the accumulated state must now be retired (start fresh) or
    ///   the old channel's history dropped or [`rekey`](RuffleState::rekey)ed, but never
    ///   silently blended.
    pub fn resume(
        configs: &[ChannelConfig],
        state: RuffleState,
        cfg: FuseConfig,
    ) -> Result<Self, ResumeError> {
        validate_registrations(configs, &cfg)?;
        validate_against_state(configs, &state, &cfg)?;
        Ok(Self {
            configs: config_lookup(configs),
            state,
            cfg,
        })
    }

    /// The persistent baseline state, for serialization and inspection.
    ///
    /// This is the object a caller persists: serialize `fuser.state()` to save, and
    /// restore through [`Fuser::resume`](Self::resume). Access is read-only by design.
    /// Every write goes through [`fuse`](Self::fuse),
    /// [`refresh_coupling`](Self::refresh_coupling), or [`resume`](Self::resume), so a
    /// caller cannot edit the `format_version`, fingerprint, or tags the merge gate
    /// relies on.
    #[must_use]
    pub fn state(&self) -> &RuffleState {
        &self.state
    }

    /// The fusion configuration in force.
    #[must_use]
    pub fn config(&self) -> &FuseConfig {
        &self.cfg
    }

    /// Fuse one query's per-channel results into a single ranking, and fold this query's
    /// readings into the running baselines. Returns the ranking, the weights used,
    /// per-channel flags, and two agreement diagnostics.
    ///
    /// The pipeline is:
    ///
    /// 1. Ensure a [`ChannelSummary`] exists for every registered channel present in
    ///    `obs`. A brand-new channel is seeded from its config: the good-score reference
    ///    from a declared [`GoodScore`](crate::score::GoodScore) (else empty), an empty
    ///    separation baseline, and the channel's tag; its declared direction is recorded
    ///    in the state fingerprint.
    /// 2. Read each channel's discrimination against its summary.
    /// 3. Assemble redundancy-discounted weights summing to `N`, the channel count.
    /// 4. Fuse by weighted reciprocal-rank fusion.
    /// 5. Flag any non-standard weighting.
    /// 6. Read the set-overlap diagnostics from the discriminating channels.
    /// 7. Update the baselines: optionally decay, then push the winsorized separation
    ///    read and the top-`m` reference read.
    ///
    /// An input whose key is not a registered channel is skipped entirely: it is excluded
    /// from discrimination, weighting, fusion, flags, and diagnostics, and never seeds or
    /// updates state. Without a registration `ruffle` has no direction, tag, or reference
    /// to interpret the channel safely, so it is ignored rather than fused at a guessed
    /// weight. If one channel key appears more than once in `obs`, only the first input
    /// is fused; a later duplicate is skipped rather than double-counting the channel's
    /// vote under a single weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use ruffle::{ChannelConfig, ChannelId, ChannelInput, Direction, FuseConfig, Fuser, Score};
    ///
    /// // A channel's native score becomes a `Score` only through a caller newtype.
    /// struct Sim(f64);
    /// impl Score for Sim {
    ///     fn value(&self) -> f64 {
    ///         self.0
    ///     }
    /// }
    ///
    /// let cfg = ChannelConfig::new(
    ///     ChannelId::new("dense", "clip-v1"),
    ///     Direction::HigherIsBetter,
    ///     None,
    /// );
    /// let mut fuser = Fuser::new(std::slice::from_ref(&cfg), FuseConfig::default())?;
    ///
    /// let obs = vec![ChannelInput::scored(&cfg, vec![(1u32, Sim(0.9)), (2, Sim(0.4))])];
    /// let fused = fuser.fuse(&obs);
    /// assert!(!fused.ranking.is_empty());
    /// # Ok::<(), ruffle::ConfigError>(())
    /// ```
    pub fn fuse<Id: Hash + Eq + Clone>(&mut self, obs: &[ChannelInput<Id>]) -> Fused<Id> {
        // Step 1: seed a summary and record the direction for every registered channel
        // present this query. The fingerprint direction is recorded once per channel and
        // never overwritten, since direction is a fixed registration fact (§4, §7).
        for o in obs {
            if let Some(ch_cfg) = self.configs.get(&o.key) {
                self.state
                    .fingerprint
                    .directions
                    .entry(o.key.clone())
                    .or_insert(ch_cfg.direction);
                if !self.state.channels.contains_key(&o.key) {
                    let summary = seed_summary(ch_cfg);
                    self.state.channels.insert(o.key.clone(), summary);
                }
            }
        }

        // Steps 2-6: the pure pipeline, reading the freshly-ensured baselines.
        let (fused, reads) = fuse_core(
            obs,
            &self.configs,
            &self.state.channels,
            &self.state.pairs,
            &self.cfg,
        );

        // Step 7: fold this query's reads into the persistent baselines. Decay first
        // (per-update cadence) so the effective count tracks recency, then winsorize the
        // separation read BEFORE pushing it so one extreme query cannot corrupt the
        // streaming mean the standardization depends on (§4, §7). Ranks-only and
        // empty-pool channels have no usable reads (both `None`), so they push nothing.
        for (key, disc) in &reads {
            if let Some(summary) = self.state.channels.get_mut(key) {
                if self.cfg.decay.enabled {
                    summary.separation.decay(self.cfg.decay.factor);
                    summary.reference.decay(self.cfg.decay.factor);
                }
                if let Some(raw) = disc.raw_separation {
                    let winsorized =
                        winsorize_separation(raw, &summary.separation, &self.cfg.discrimination);
                    summary.separation.push(winsorized);
                }
                if let Some(top) = disc.top_m_average {
                    summary.reference.push(top);
                }
            }
        }

        fused
    }

    /// Fuse one query against the given configs and a prior state, without mutating any
    /// baseline. Without a usable prior it reduces to within-query reciprocal-rank fusion.
    ///
    /// This runs the same weighting and fusion as [`fuse`](Self::fuse) but seeds nothing
    /// into shared state and updates no baseline. Each registered channel present in `obs`
    /// standardizes against `prior.channels` if it appears there, otherwise against a
    /// summary freshly seeded from its config (including the good-score reference, if
    /// declared). With an empty prior and no declared references, the separation reads
    /// neutralize and the absolute-goodness term is unavailable, so every weight lands at
    /// the neutral `1.0` and the fusion reduces to standard, unweighted RRF.
    ///
    /// # Errors
    ///
    /// Runs the same gates as [`resume`](Self::resume): the registrations and
    /// configuration must be valid, and the prior must be compatible with them. A
    /// mismatched prior would standardize this query against baselines measured under a
    /// different model or orientation; the fusion is read-only, but its weights would be
    /// corrupt all the same.
    pub fn fuse_stateless<Id: Hash + Eq + Clone>(
        obs: &[ChannelInput<Id>],
        configs: &[ChannelConfig],
        prior: &RuffleState,
        cfg: &FuseConfig,
    ) -> Result<Fused<Id>, ResumeError> {
        validate_registrations(configs, cfg)?;
        validate_against_state(configs, prior, cfg)?;
        // Build the per-channel lookup internally, keyed by join handle, so the caller
        // passes a plain slice of configs.
        let configs = config_lookup(configs);
        // Build the baselines to standardize against, locally and without mutating the
        // prior: the prior summary where present, else a fresh seed from the config.
        let mut channels: BTreeMap<String, ChannelSummary> = BTreeMap::new();
        for o in obs {
            if let Some(ch_cfg) = configs.get(&o.key) {
                if !channels.contains_key(&o.key) {
                    let summary = prior
                        .channels
                        .get(&o.key)
                        .cloned()
                        .unwrap_or_else(|| seed_summary(ch_cfg));
                    channels.insert(o.key.clone(), summary);
                }
            }
        }
        let (fused, _reads) = fuse_core(obs, &configs, &channels, &prior.pairs, cfg);
        Ok(fused)
    }

    /// Fold a full-scored anchor's pairwise correlations into the persistent redundancy
    /// baselines.
    ///
    /// Each pair's correlation is accumulated into its persistent redundancy summary by
    /// merging in a reading of `(mean = correlation, variance = 0, count = n_both)`. The
    /// accumulation gives the summary three quantities at once: `redundancy.count()` is
    /// the total both-scored overlap (the reliability [`coupled_weights`] gates the
    /// discount on), `redundancy.mean()` is the overlap-weighted pooled correlation (the
    /// point estimate), and `redundancy.variance()` is the variability across refreshes
    /// and strata (which [`coupled_weights`] uses to drop a pair whose redundancy is
    /// unstable). When decay is enabled the existing pair is decayed first, so an old
    /// correlation fades as fresh anchors arrive.
    pub fn refresh_coupling(&mut self, anchor: &Anchor) {
        let obs_corr = anchor_correlations(anchor, &self.cfg.coupling);
        for (pair, observation) in obs_corr {
            let entry = self.state.pairs.entry(pair).or_default();
            if self.cfg.decay.enabled {
                entry.redundancy.decay(self.cfg.decay.factor);
                entry.refreshes *= decay_factor(self.cfg.decay.factor);
            }
            let reading =
                MeanVar::from_prior(observation.correlation, 0.0, observation.n_both as f64);
            entry.redundancy.merge_in(&reading);
            // One refresh contributed one between-refresh observation, whatever its
            // overlap: the refresh count is what the §5.3 stability gate is denominated
            // in.
            entry.refreshes += 1.0;
        }
    }
}

/// Validate the registrations and configuration on their own (§4, §7, §8): every knob in
/// range, every join-handle key distinct, and every declared good score orientable to a
/// usable reference.
fn validate_registrations(configs: &[ChannelConfig], cfg: &FuseConfig) -> Result<(), ConfigError> {
    cfg.validate()?;
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for c in configs {
        if !seen.insert(c.id.key.as_str()) {
            return Err(ConfigError::DuplicateChannelKey {
                key: c.id.key.clone(),
            });
        }
        if let Some(good_score) = c.good_score {
            if good_score.oriented(c.direction).is_none() {
                return Err(ConfigError::InvalidGoodScore {
                    channel: c.id.key.clone(),
                    reason: "anchors must be finite and `good` must exceed `typical` \
                             after orientation",
                });
            }
        }
    }
    Ok(())
}

/// Validate the registrations against a persisted state (§8): the same compatibility
/// gate [`RuffleState::merge`] runs, applied at the live resume boundary.
fn validate_against_state(
    configs: &[ChannelConfig],
    state: &RuffleState,
    cfg: &FuseConfig,
) -> Result<(), Mismatch> {
    if state.format_version() != RuffleState::FORMAT_VERSION {
        return Err(Mismatch::FormatVersion {
            left: RuffleState::FORMAT_VERSION,
            right: state.format_version(),
        });
    }
    if state.fingerprint().stat_version != StatFingerprint::STAT_VERSION
        || state.fingerprint().baseline_mode != cfg.baseline_mode
    {
        return Err(Mismatch::Fingerprint);
    }
    for c in configs {
        if let Some(dir) = state.fingerprint().directions.get(&c.id.key) {
            if *dir != c.direction {
                return Err(Mismatch::DirectionConflict {
                    channel: c.id.key.clone(),
                });
            }
        }
        if let Some(summary) = state.channels.get(&c.id.key) {
            if summary.tag != c.id.tag {
                return Err(Mismatch::Tag {
                    channel: c.id.key.clone(),
                    left: summary.tag.clone(),
                    right: c.id.tag.clone(),
                });
            }
        }
    }
    Ok(())
}

/// Build the per-channel lookup keyed by each config's join handle `id.key` (§11).
///
/// Key distinctness is enforced by [`validate_registrations`] before any lookup is
/// built, so the map is total over the registrations.
fn config_lookup(configs: &[ChannelConfig]) -> BTreeMap<String, ChannelConfig> {
    configs
        .iter()
        .map(|c| (c.id.key.clone(), c.clone()))
        .collect()
}

/// Seed a fresh per-channel summary from its registration (§4, §8).
///
/// The good-score reference is seeded from a declared [`GoodScore`](crate::score::GoodScore),
/// oriented to canonical higher-is-better, as `from_prior(mu_ref, sigma_ref², n0)`; when
/// no good score is declared (or it is degenerate) the reference starts empty and `D^abs`
/// cold-starts. The separation baseline always starts empty. The tag comes from the
/// config and gates every later merge.
fn seed_summary(cfg: &ChannelConfig) -> ChannelSummary {
    let reference = match cfg.good_score {
        Some(good_score) => match good_score.oriented(cfg.direction) {
            Some(reference) => MeanVar::from_prior(
                reference.mu_ref,
                reference.sigma_ref * reference.sigma_ref,
                good_score.weight,
            ),
            None => MeanVar::new(),
        },
        None => MeanVar::new(),
    };
    ChannelSummary::with_reference(cfg.id.tag.clone(), reference)
}

/// The pure heart of fusion shared by [`Fuser::fuse`] and [`Fuser::fuse_stateless`]
/// (steps 2-6 of §11): read discrimination, weight, fuse, flag, and diagnose, reading
/// baselines from `channels` and `pairs` and mutating nothing.
///
/// Returns the [`Fused`] result plus the per-channel discrimination reads, which the
/// stateful caller folds back into the baselines (step 7). An input whose key is
/// not in `configs`, or that has no summary in `channels`, is skipped entirely.
fn fuse_core<Id: Hash + Eq + Clone>(
    obs: &[ChannelInput<Id>],
    configs: &BTreeMap<String, ChannelConfig>,
    channels: &BTreeMap<String, ChannelSummary>,
    pairs: &BTreeMap<UnorderedPair<String>, PairSummary>,
    cfg: &FuseConfig,
) -> (Fused<Id>, BTreeMap<String, ChannelDiscrimination>) {
    // Steps 2 and 5: discrimination and flags over the registered channels present this
    // query. `g_map` and `reads` are keyed by channel; only the FIRST input per key is
    // read and fused (a later duplicate would double-count the channel's vote under a
    // single weight). `registered` holds the inputs that actually fuse.
    let mut g_map: BTreeMap<String, f64> = BTreeMap::new();
    let mut reads: BTreeMap<String, ChannelDiscrimination> = BTreeMap::new();
    let mut flags: BTreeMap<String, ChannelFlag> = BTreeMap::new();
    let mut registered: Vec<&ChannelInput<Id>> = Vec::new();
    // Step 6 collects the discriminating channels' top sets as it goes (§5.5).
    let mut top_sets: Vec<(String, Vec<Id>)> = Vec::new();

    for o in obs {
        if configs.get(&o.key).is_none() {
            continue; // unregistered: skip entirely (no direction/tag/reference)
        }
        if reads.contains_key(&o.key) {
            continue; // duplicate channel key: first input wins, never double-fused
        }
        let Some(summary) = channels.get(&o.key) else {
            continue; // no baseline to standardize against; defensively skip
        };
        // Project the §8 summary down to the two bare baselines the pure §4 estimator
        // reads; the Fuser, which holds both layers, does the projection.
        let disc = discriminate(
            &o.items,
            &summary.separation,
            &summary.reference,
            &cfg.discrimination,
        );

        // At most one flag per channel (the output type carries one). Ranks-only has no
        // score statistic at all and outranks the two scored conditions; between the
        // scored conditions, a degenerate separation (the primary "can it rank" read)
        // outranks a cold reference (an often-known cold-start state).
        if o.items.is_ranks_only() {
            flags.insert(o.key.clone(), ChannelFlag::RanksOnlyDefaultWeighted);
        } else if disc.degenerate_separation {
            flags.insert(o.key.clone(), ChannelFlag::DegenerateSeparation);
        } else if disc.reference_cold {
            flags.insert(o.key.clone(), ChannelFlag::NoReference);
        }

        // Step 6 membership: a channel is discriminating when it produced an actual
        // separation read (which excludes ranks-only, empty, and degenerate pools), is
        // performing at or above its own norm, and that norm is backed by enough
        // baseline observations to mean something. An empty pool must not enter with an
        // empty top set (it would zero the intersection and assert maximal conflict on
        // any query where one facet legitimately found nothing), and a stone-cold
        // channel's neutral 1.0 is absence of evidence, not evidence of discrimination
        // -- fewer than two qualifying channels reads as "no signal" (0, 0), the honest
        // cold-start diagnostic (§5.5).
        if disc.raw_separation.is_some()
            && disc.g >= NEUTRAL_WEIGHT
            && summary.separation.count() >= cfg.discrimination.min_count_for_z
        {
            top_sets.push((o.key.clone(), top_m_ids(&o.items, cfg.discrimination.top_m)));
        }

        g_map.insert(o.key.clone(), disc.g);
        reads.insert(o.key.clone(), disc);
        registered.push(o);
    }

    // Step 3: redundancy-discounted weights over the present registered channels. The
    // keys are sorted (BTreeMap order), so the weighting is deterministic and the weights
    // sum to `N` = the number of those channels (§5.4, §6).
    let keys: Vec<String> = g_map.keys().cloned().collect();
    // Project the §8 pair summaries down to the bare redundancy baselines the pure §5
    // estimator consumes; `PairBaseline` is `Copy`-cheap, so the projection is cheap.
    let redundancy: BTreeMap<UnorderedPair<String>, PairBaseline> = pairs
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                PairBaseline {
                    redundancy: v.redundancy,
                    refreshes: v.refreshes,
                },
            )
        })
        .collect();
    let coupled = coupled_weights(&g_map, &redundancy, &keys, &cfg.coupling);

    // Step 4: fuse by weighted RRF. Fuse only the registered inputs; in the common
    // case every input is registered and the input slice is reused as-is, so the
    // potentially-large score vectors are not cloned.
    let ranking = if registered.len() == obs.len() {
        weighted_rrf(obs, &coupled.weights, &cfg.fusion)
    } else {
        let filtered: Vec<ChannelInput<Id>> = registered.iter().map(|o| (*o).clone()).collect();
        weighted_rrf(&filtered, &coupled.weights, &cfg.fusion)
    };

    // Step 6: diagnostics from set membership over the discriminating channels (§5.5),
    // collected in the loop above. Fewer than two qualifying channels yields (0, 0)
    // inside `diagnostics`.
    let Diagnostics {
        confidence,
        conflict,
    } = diagnostics(&top_sets);

    let fused = Fused {
        ranking,
        weights: coupled.weights,
        flags,
        discrimination: reads.clone(),
        confidence,
        conflict,
    };
    (fused, reads)
}

/// The channel's top-`m` candidate ids: by descending oriented score for a `Scored`
/// channel (ties broken by list order, a total order, so membership is deterministic),
/// by list order for a `Ranks` channel. Used to build the diagnostic top-set overlap
/// (§5.5), which reads membership only, so the returned order is unspecified. Selection
/// is `O(n)` rather than a full sort.
fn top_m_ids<Id: Clone>(items: &Items<Id>, m: usize) -> Vec<Id> {
    match items {
        Items::Scored(v) => {
            let mut order: Vec<usize> = (0..v.len()).collect();
            let by_score_desc = |a: &usize, b: &usize| {
                v[*b]
                    .1
                    .partial_cmp(&v[*a].1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.cmp(b))
            };
            if m < order.len() {
                order.select_nth_unstable_by(m, by_score_desc);
                order.truncate(m);
            }
            order.into_iter().map(|i| v[i].0.clone()).collect()
        }
        Items::Ranks(v) => v.iter().take(m).cloned().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CouplingConfig, DecayConfig, DiscriminationConfig};
    use crate::keys::{BaselineMode, ChannelId, StatFingerprint};
    use crate::score::{Direction, GoodScore, Score};
    use crate::weighting::discrimination::discriminate;
    use approx::assert_abs_diff_eq;

    /// A caller-side newtype: the only way a bare number becomes a [`Score`] (§7).
    struct Val(f64);
    impl Score for Val {
        fn value(&self) -> f64 {
            self.0
        }
    }

    fn key(s: &str) -> String {
        s.to_string()
    }

    fn tag() -> String {
        "t".to_string()
    }

    fn chan(name: &str, good: Option<GoodScore>) -> ChannelConfig {
        ChannelConfig::new(ChannelId::new(name, tag()), Direction::HigherIsBetter, good)
    }

    /// A pool with 30 distinct bulk values in `[0, 3)` and two spikes far above, so the
    /// separation statistic is well-defined (≥ 8 distinct values, a clear top elevation).
    /// `base` offsets the ids so several channels can share or disjoin their pools.
    fn spiked(base: u32) -> Vec<(u32, Val)> {
        let mut v: Vec<(u32, Val)> = (0..30).map(|i| (base + i, Val(i as f64 * 0.1))).collect();
        v.push((base + 100, Val(10.0)));
        v.push((base + 101, Val(10.5)));
        v
    }

    fn scored_obs(cfg: &ChannelConfig, base: u32) -> ChannelInput<u32> {
        ChannelInput::scored(cfg, spiked(base))
    }

    fn ranks_obs(cfg: &ChannelConfig, ids: &[u32]) -> ChannelInput<u32> {
        ChannelInput::ranked(cfg, ids.to_vec())
    }

    /// A `MeanVar` seeded to a chosen mean, variance, and count (a stand-in for a
    /// baseline already accumulated from traffic).
    fn baseline(mean: f64, variance: f64, count: f64) -> MeanVar {
        MeanVar::from_prior(mean, variance, count)
    }

    fn empty_state() -> RuffleState {
        RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, BTreeMap::new()))
    }

    fn fuser(cfgs: &[ChannelConfig], cfg: FuseConfig) -> Fuser {
        Fuser::new(cfgs, cfg).expect("valid registrations")
    }

    /// The raw separation a pool reads against a cold baseline (count 0 still computes it).
    fn raw_sep(base: u32) -> f64 {
        discriminate(
            &ChannelInput::scored(&chan("x", None), spiked(base)).items,
            &MeanVar::new(),
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        )
        .raw_separation
        .expect("separation defined for the spiked pool")
    }

    // --- 1. end-to-end fuse over three mixed channels ---------------------------------

    #[test]
    fn end_to_end_three_channels_mixed() {
        let a = chan("a", None);
        let b = chan("b", None);
        let r = chan("r", None);
        let mut f = fuser(&[a.clone(), b.clone(), r.clone()], FuseConfig::default());

        let obs = vec![
            scored_obs(&a, 0),
            scored_obs(&b, 0), // share ids with a, so the top sets overlap
            ranks_obs(&r, &[100, 101, 5, 6, 7]),
        ];
        let fused = f.fuse(&obs);

        // A ranking came out, covering the surfaced ids.
        assert!(!fused.ranking.is_empty());
        assert!(fused.ranking.iter().any(|(id, _)| *id == 100));

        // Weights sum to N = 3 and are all non-negative (§9 invariant 9).
        let total: f64 = fused.weights.values().sum();
        assert_abs_diff_eq!(total, 3.0, epsilon = 1e-9);
        assert!(fused.weights.values().all(|&w| w >= 0.0));

        // The ranks-only channel is flagged and carried at its default weight; the
        // scored channels are not flagged ranks-only.
        assert_eq!(
            fused.flags.get(&key("r")),
            Some(&ChannelFlag::RanksOnlyDefaultWeighted)
        );
        assert_ne!(
            fused.flags.get(&key("a")),
            Some(&ChannelFlag::RanksOnlyDefaultWeighted)
        );

        // The discrimination reads behind the weights are surfaced per fused channel.
        assert_eq!(fused.discrimination.len(), 3);
        assert!(fused.discrimination[&key("a")].raw_separation.is_some());
        assert_eq!(fused.discrimination[&key("r")].raw_separation, None);

        // Every baseline is stone-cold, so no channel qualifies as discriminating and
        // the diagnostics stay at the honest "no signal" (0, 0) rather than asserting
        // confidence from a first sight of data (§5.5).
        assert_eq!(fused.confidence, 0.0);
        assert_eq!(fused.conflict, 0.0);
    }

    // --- 1b. diagnostics fire only for evidence-backed channels ------------------------

    /// A state whose channels a and b carry warm separation baselines seeded just below
    /// the given raw read, so both standardize positive (g >= 1) and qualify as
    /// discriminating.
    fn warm_two_channel_state(r: f64) -> RuffleState {
        let mut state = empty_state();
        for name in ["a", "b"] {
            state.channels.insert(
                key(name),
                ChannelSummary {
                    separation: baseline(r - 1.0, 1.0, 6.0),
                    reference: MeanVar::new(),
                    tag: tag(),
                },
            );
        }
        state
    }

    #[test]
    fn diagnostics_fire_once_baselines_are_backed() {
        let a = chan("a", None);
        let b = chan("b", None);
        let state = warm_two_channel_state(raw_sep(0));
        let mut f = Fuser::resume(&[a.clone(), b.clone()], state, FuseConfig::default()).unwrap();

        // Identical pools: the discriminating channels agree completely.
        let fused = f.fuse(&[scored_obs(&a, 0), scored_obs(&b, 0)]);
        assert_abs_diff_eq!(fused.confidence, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fused.conflict, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fused.confidence + fused.conflict, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn empty_channel_does_not_corrupt_diagnostics() {
        // A registered channel that surfaced NOTHING must not enter the top-set overlap:
        // its empty set would zero the intersection and assert maximal conflict on a
        // query where two live channels agree perfectly (§5.5). Common and legitimate:
        // an image facet finding no matches for a text-only query.
        let a = chan("a", None);
        let b = chan("b", None);
        let c = chan("c", None);
        let mut state = warm_two_channel_state(raw_sep(0));
        state.channels.insert(
            key("c"),
            ChannelSummary {
                separation: baseline(2.0, 1.0, 6.0),
                reference: MeanVar::new(),
                tag: tag(),
            },
        );
        let mut f = Fuser::resume(
            &[a.clone(), b.clone(), c.clone()],
            state,
            FuseConfig::default(),
        )
        .unwrap();

        let empty: Vec<(u32, Val)> = Vec::new();
        let fused = f.fuse(&[
            scored_obs(&a, 0),
            scored_obs(&b, 0),
            ChannelInput::scored(&c, empty),
        ]);
        assert_abs_diff_eq!(fused.confidence, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fused.conflict, 0.0, epsilon = 1e-12);
    }

    /// The raw separation of an arbitrary scored pool against cold baselines.
    fn raw_sep_of_pool(pool: &[(u32, f64)]) -> f64 {
        let items: Items<u32> = Items::Scored(pool.to_vec());
        discriminate(
            &items,
            &MeanVar::new(),
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        )
        .raw_separation
        .expect("separation defined for this pool")
    }

    /// A state whose named channels carry warm separation baselines seeded just below
    /// `r`, so each standardizes positive (g >= 1) and qualifies for diagnostics.
    fn warm_state_for(names: &[&str], r: f64) -> RuffleState {
        let mut state = empty_state();
        for name in names {
            state.channels.insert(
                key(name),
                ChannelSummary {
                    separation: baseline(r - 1.0, 1.0, 6.0),
                    reference: MeanVar::new(),
                    tag: tag(),
                },
            );
        }
        state
    }

    #[test]
    fn diagnostic_top_sets_are_truncated_to_top_m() {
        // Two channels agree exactly on their 10 best-scored items but have disjoint
        // 22-item tails. The diagnostic overlap is over the top-m sets (m = top_m = 10),
        // so confidence must read 1.0; feeding whole pools into the overlap instead
        // would dilute the intersection with the disjoint tails.
        let spikes = |()| (0..10u32).map(|i| (100 + i, 10.0 + 0.05 * i as f64));
        let mut pool_a: Vec<(u32, f64)> = (0..22u32).map(|i| (i, 0.01 * i as f64)).collect();
        pool_a.extend(spikes(()));
        let mut pool_b: Vec<(u32, f64)> = (0..22u32).map(|i| (200 + i, 0.01 * i as f64)).collect();
        pool_b.extend(spikes(()));

        let a = chan("a", None);
        let b = chan("b", None);
        let state = warm_state_for(&["a", "b"], raw_sep_of_pool(&pool_a));
        let mut f = Fuser::resume(&[a.clone(), b.clone()], state, FuseConfig::default()).unwrap();

        let fused = f.fuse(&[
            ChannelInput {
                key: key("a"),
                items: Items::Scored(pool_a),
            },
            ChannelInput {
                key: key("b"),
                items: Items::Scored(pool_b),
            },
        ]);
        assert_abs_diff_eq!(fused.confidence, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn diagnostic_pool_exactly_at_top_m_reads_whole_pool() {
        // A qualifying pool of EXACTLY top_m items: the top-m set is the whole pool, no
        // selection needed, and the fuse must not panic on the boundary (an off-by-one
        // in the selection guard indexes out of bounds here).
        let pool: Vec<(u32, f64)> = (0..10u32).map(|i| (i, i as f64)).collect();
        assert_eq!(pool.len(), DiscriminationConfig::default().top_m);

        let a = chan("a", None);
        let b = chan("b", None);
        let state = warm_state_for(&["a", "b"], raw_sep_of_pool(&pool));
        let mut f = Fuser::resume(&[a.clone(), b.clone()], state, FuseConfig::default()).unwrap();

        let fused = f.fuse(&[
            ChannelInput {
                key: key("a"),
                items: Items::Scored(pool.clone()),
            },
            ChannelInput {
                key: key("b"),
                items: Items::Scored(pool),
            },
        ]);
        assert_abs_diff_eq!(fused.confidence, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn duplicate_channel_input_is_fused_once() {
        // The same key twice in one fuse call must not double-count the channel's vote:
        // the first input wins and the second is skipped entirely.
        let a = chan("a", None);
        let b = chan("b", None);
        let mut f1 = fuser(&[a.clone(), b.clone()], FuseConfig::default());
        let mut f2 = fuser(&[a.clone(), b.clone()], FuseConfig::default());

        let once = f1.fuse(&[scored_obs(&a, 0), scored_obs(&b, 0)]);
        let duped = f2.fuse(&[scored_obs(&a, 0), scored_obs(&b, 0), scored_obs(&a, 0)]);
        assert_eq!(once.ranking, duped.ranking);
        assert_eq!(once.weights, duped.weights);
    }

    // --- 2. the stateful update moves the baselines -----------------------------------

    #[test]
    fn fuse_grows_baselines_and_restandardizes() {
        // Channel a: scored, no declared reference (so it learns one from traffic), with a
        // separation baseline pre-seeded one z below the pool's read, so its standardized
        // separation is non-trivial and shifts as the baseline absorbs the read. Channel
        // b: ranks-only, so its weight is fixed at the default and a's weight tracks a's
        // own discrimination alone.
        let a = chan("a", None);
        let b = chan("b", None);
        let r = raw_sep(0);

        let mut state = empty_state();
        state.channels.insert(
            key("a"),
            ChannelSummary {
                separation: baseline(r - 1.5, 1.0, 6.0), // z(r) = (r − (r−1.5)) / 1 = 1.5
                reference: MeanVar::new(),
                tag: tag(),
            },
        );
        let mut f = Fuser::resume(&[a.clone(), b.clone()], state, FuseConfig::default()).unwrap();

        let obs = vec![scored_obs(&a, 0), ranks_obs(&b, &[100, 101, 1, 2])];

        let fused1 = f.fuse(&obs);
        let w_a1 = fused1.weights[&key("a")];
        assert_eq!(f.state.channels[&key("a")].separation.count(), 7.0); // 6 → 7
        assert_eq!(f.state.channels[&key("a")].reference.count(), 1.0); // learned: 0 → 1
        let ref_mean1 = f.state.channels[&key("a")].reference.mean();
        assert!(ref_mean1.is_finite() && ref_mean1 > 0.0); // refined from the observed top

        let fused2 = f.fuse(&obs);
        let w_a2 = fused2.weights[&key("a")];
        assert_eq!(f.state.channels[&key("a")].separation.count(), 8.0); // 7 → 8
        assert_eq!(f.state.channels[&key("a")].reference.count(), 2.0); // 1 → 2

        // The separation baseline moved between the two fuses, so the same raw read
        // standardizes differently and the channel's weight changes.
        assert!(
            (w_a1 - w_a2).abs() > 1e-6,
            "weight should shift as the baseline moves: {w_a1} vs {w_a2}"
        );
        assert_eq!(
            fused1.flags.get(&key("b")),
            Some(&ChannelFlag::RanksOnlyDefaultWeighted)
        );
    }

    // --- 3. winsorize before the baseline update --------------------------------------

    #[test]
    fn extreme_separation_read_is_winsorized_before_update() {
        // A pool whose raw separation sits far above the seeded baseline. Winsorizing it
        // before the push keeps it from dragging the streaming mean the way the raw value
        // would (§4, §7).
        let a = chan("a", None);
        let r = raw_sep(0);

        // Seed a tight baseline well below r: mean 1.0, std 0.1, count 6. The winsor band
        // is mean ± 4·std = [0.6, 1.4], and r (a large ratio) lands far outside it.
        let seed = baseline(1.0, 0.01, 6.0);
        let winsorized = winsorize_separation(r, &seed, &DiscriminationConfig::default());
        assert!(winsorized < r, "the extreme read must actually be clamped");

        // The two counterfactual baseline means: pushing the clamped value vs the raw one.
        let mut clamped_push = seed;
        clamped_push.push(winsorized);
        let mut raw_push = seed;
        raw_push.push(r);

        let mut state = empty_state();
        state.channels.insert(
            key("a"),
            ChannelSummary {
                separation: seed,
                reference: MeanVar::new(),
                tag: tag(),
            },
        );
        let mut f = Fuser::resume(std::slice::from_ref(&a), state, FuseConfig::default()).unwrap();
        f.fuse(&[scored_obs(&a, 0)]);

        let moved = f.state.channels[&key("a")].separation.mean();
        // The fuser pushed the winsorized value, not the raw one.
        assert_abs_diff_eq!(moved, clamped_push.mean(), epsilon = 1e-12);
        assert!(
            moved < raw_push.mean(),
            "winsorize limited the move: {moved} should be below the raw-push mean {}",
            raw_push.mean()
        );
    }

    // --- 4. decay slows the baseline growth -------------------------------------------

    #[test]
    fn decay_slows_baseline_count_growth() {
        let a = chan("a", None);
        let obs = vec![scored_obs(&a, 0)];

        let mut plain = fuser(std::slice::from_ref(&a), FuseConfig::default());
        let mut decayed = fuser(
            std::slice::from_ref(&a),
            FuseConfig {
                decay: DecayConfig {
                    enabled: true,
                    factor: 0.9,
                },
                ..FuseConfig::default()
            },
        );

        for _ in 0..10 {
            plain.fuse(&obs);
            decayed.fuse(&obs);
        }

        let plain_count = plain.state.channels[&key("a")].separation.count();
        let decayed_count = decayed.state.channels[&key("a")].separation.count();
        assert_abs_diff_eq!(plain_count, 10.0, epsilon = 1e-9); // one push per fuse
        assert!(
            decayed_count < plain_count,
            "decay-then-push must grow the count more slowly: {decayed_count} vs {plain_count}"
        );
    }

    // --- 5. fuse_stateless degrades to RRF, weights per the prior ----------------------

    #[test]
    fn stateless_empty_prior_is_unweighted_rrf() {
        // No prior, no declared references: every channel reads neutral, so the weights
        // are all 1.0 and the fusion is standard, unweighted RRF.
        let a = chan("a", None);
        let b = chan("b", None);
        let cfgs = [a.clone(), b.clone()];
        let prior = empty_state();
        let obs = vec![scored_obs(&a, 0), scored_obs(&b, 0)];

        let fused = Fuser::fuse_stateless(&obs, &cfgs, &prior, &FuseConfig::default()).unwrap();
        for w in fused.weights.values() {
            assert_abs_diff_eq!(*w, 1.0, epsilon = 1e-9);
        }
        // With equal weights the result matches a direct unweighted RRF call.
        let direct = weighted_rrf(&obs, &fused.weights, &FuseConfig::default().fusion);
        assert_eq!(fused.ranking, direct);
    }

    #[test]
    fn stateless_rich_prior_tilts_weights() {
        // Channel x carries a separation baseline below the pool read (so it discriminates
        // unusually well here); channel y is cold. The prior tilts weight toward x.
        let x = chan("x", None);
        let y = chan("y", None);
        let cfgs = [x.clone(), y.clone()];
        let r = raw_sep(0);

        let mut prior = empty_state();
        prior.channels.insert(
            key("x"),
            ChannelSummary {
                separation: baseline(r - 1.5, 1.0, 8.0),
                reference: MeanVar::new(),
                tag: tag(),
            },
        );
        // y left absent from the prior: it seeds cold from its config.

        let obs = vec![scored_obs(&x, 0), scored_obs(&y, 0)];
        let fused = Fuser::fuse_stateless(&obs, &cfgs, &prior, &FuseConfig::default()).unwrap();

        let total: f64 = fused.weights.values().sum();
        assert_abs_diff_eq!(total, 2.0, epsilon = 1e-9);
        assert!(
            fused.weights[&key("x")] > fused.weights[&key("y")],
            "the discriminating channel should carry more weight: {:?}",
            fused.weights
        );
    }

    #[test]
    fn stateless_does_not_mutate_the_prior() {
        let a = chan("a", Some(GoodScore::new(0.0, 1.0, 5.0)));
        let cfgs = [a.clone()];
        let mut prior = empty_state();
        prior.channels.insert(
            key("a"),
            ChannelSummary {
                separation: baseline(2.0, 1.0, 6.0),
                reference: baseline(0.0, 0.25, 5.0),
                tag: tag(),
            },
        );
        let before = prior.clone();
        let _ = Fuser::fuse_stateless(&[scored_obs(&a, 0)], &cfgs, &prior, &FuseConfig::default())
            .unwrap();
        assert_eq!(prior, before);
    }

    // --- 7. refresh_coupling accumulates reliability, mean, and variance --------------

    /// A two-channel full-scored anchor: channel `a` = id, channel `b` = the closure.
    fn anchor_of(n: u32, b: impl Fn(u32) -> f64) -> Anchor {
        let cands: Vec<u32> = (0..n).collect();
        let ca = chan("a", None);
        let cb = chan("b", None);
        Anchor::build(&cands, &[&ca, &cb], move |id, k| {
            if k == "a" {
                Some(Val(*id as f64))
            } else {
                Some(Val(b(*id)))
            }
        })
    }

    #[test]
    fn refresh_coupling_accumulates_overlap_and_mean() {
        let pair = UnorderedPair::new(key("a"), key("b"));
        let mut f = fuser(&[chan("a", None), chan("b", None)], FuseConfig::default());

        // b = a → rank correlation +1 over the 40 both-scored items.
        let anchor = anchor_of(40, |id| id as f64);
        f.refresh_coupling(&anchor);
        let red = &f.state.pairs[&pair].redundancy;
        assert_eq!(red.count(), 40.0);
        assert_abs_diff_eq!(red.mean(), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(red.variance(), 0.0, epsilon = 1e-12);
        assert_eq!(f.state.pairs[&pair].refreshes, 1.0);

        // A second identical refresh: reliability doubles, mean holds, variance stays ~0,
        // and the refresh count now clears the default min_refreshes gate of 2.
        f.refresh_coupling(&anchor);
        let red = &f.state.pairs[&pair].redundancy;
        assert_eq!(red.count(), 80.0);
        assert_abs_diff_eq!(red.mean(), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(red.variance(), 0.0, epsilon = 1e-12);
        assert_eq!(f.state.pairs[&pair].refreshes, 2.0);
    }

    #[test]
    fn refresh_coupling_raises_variance_across_differing_strata() {
        let pair = UnorderedPair::new(key("a"), key("b"));
        let mut f = fuser(&[chan("a", None), chan("b", None)], FuseConfig::default());

        // First stratum: b = a → +1. Second: b = −a → −1. The pooled summary now carries
        // a between-refresh variance, the §5.3 stratum-stability signal.
        f.refresh_coupling(&anchor_of(40, |id| id as f64));
        f.refresh_coupling(&anchor_of(40, |id| -(id as f64)));
        let red = &f.state.pairs[&pair].redundancy;
        assert_eq!(red.count(), 80.0);
        assert_abs_diff_eq!(red.mean(), 0.0, epsilon = 1e-9); // +1 and −1 pool to 0
        assert!(red.variance() > 0.5, "variance: {}", red.variance());
    }

    // --- 8. construction and resume gates ----------------------------------------------

    #[test]
    fn new_refuses_invalid_config_duplicate_key_and_bad_good_score() {
        // Inverted g bounds: rejected before any query can hit them.
        let mut bad = FuseConfig::default();
        bad.discrimination.g_floor = 5.0;
        bad.discrimination.g_upper_bound = 4.0;
        assert!(matches!(
            Fuser::new(&[chan("a", None)], bad),
            Err(ConfigError::InvalidFuseConfig { .. })
        ));

        // Duplicate join-handle key: both registrations would write one baseline.
        assert!(matches!(
            Fuser::new(&[chan("a", None), chan("a", None)], FuseConfig::default()),
            Err(ConfigError::DuplicateChannelKey { .. })
        ));

        // A good score that cannot orient (good <= typical): refused loudly instead of
        // silently cold-starting the channel as if nothing had been declared.
        let bad_ref = chan("a", Some(GoodScore::new(0.5, 0.3, 4.0)));
        assert!(matches!(
            Fuser::new(&[bad_ref], FuseConfig::default()),
            Err(ConfigError::InvalidGoodScore { .. })
        ));
    }

    #[test]
    fn resume_refuses_a_bumped_tag() {
        // The live path of the §8 tag gate: a model swap crosses a restart, so resume
        // must refuse to continue a v2-tagged channel on baselines accumulated under v1.
        // (Before this gate, fuse() silently kept accumulating into the v1 baseline.)
        let v1 = ChannelConfig::new(
            ChannelId::new("a", "model-v1"),
            Direction::HigherIsBetter,
            None,
        );
        let mut f = fuser(std::slice::from_ref(&v1), FuseConfig::default());
        f.fuse(&[scored_obs(&v1, 0)]);
        let state = f.state().clone();

        let v2 = ChannelConfig::new(
            ChannelId::new("a", "model-v2"),
            Direction::HigherIsBetter,
            None,
        );
        let err = Fuser::resume(std::slice::from_ref(&v2), state, FuseConfig::default())
            .expect_err("a bumped tag over old baselines must refuse");
        assert!(matches!(
            err,
            ResumeError::State(Mismatch::Tag { ref channel, ref left, ref right })
                if channel == "a" && left == "model-v1" && right == "model-v2"
        ));
    }

    #[test]
    fn resume_refuses_a_direction_flip() {
        let hi = ChannelConfig::new(ChannelId::new("a", "t"), Direction::HigherIsBetter, None);
        let f = fuser(std::slice::from_ref(&hi), FuseConfig::default());
        let state = f.state().clone();

        let lo = ChannelConfig::new(ChannelId::new("a", "t"), Direction::LowerIsBetter, None);
        let err = Fuser::resume(std::slice::from_ref(&lo), state, FuseConfig::default())
            .expect_err("a flipped direction must refuse");
        assert!(matches!(
            err,
            ResumeError::State(Mismatch::DirectionConflict { ref channel }) if channel == "a"
        ));
    }

    #[test]
    fn resume_refuses_a_foreign_version_or_baseline_mode() {
        let a = chan("a", None);

        // A state at another format version: refused with the build's version on the left.
        let stale = {
            let mut value = serde_json::to_value(empty_state()).unwrap();
            value["format_version"] = serde_json::Value::from(99u32);
            serde_json::from_value::<RuffleState>(value).unwrap()
        };
        let err = Fuser::resume(std::slice::from_ref(&a), stale, FuseConfig::default())
            .expect_err("a foreign format version must refuse");
        assert!(matches!(
            err,
            ResumeError::State(Mismatch::FormatVersion {
                left: RuffleState::FORMAT_VERSION,
                right: 99
            })
        ));

        // A state at another statistic version: numerically incompatible summaries.
        let stale_stat = {
            let mut value = serde_json::to_value(empty_state()).unwrap();
            value["fingerprint"]["stat_version"] = serde_json::Value::from(1u32);
            serde_json::from_value::<RuffleState>(value).unwrap()
        };
        let err = Fuser::resume(std::slice::from_ref(&a), stale_stat, FuseConfig::default())
            .expect_err("a foreign statistic version must refuse");
        assert!(matches!(err, ResumeError::State(Mismatch::Fingerprint)));
    }

    #[test]
    fn resume_accepts_a_matching_state_round_trip() {
        // The everyday save/restart/resume cycle stays frictionless: same configs, same
        // tags, same directions -> resume succeeds and continues accumulating.
        let a = chan("a", None);
        let mut f = fuser(std::slice::from_ref(&a), FuseConfig::default());
        f.fuse(&[scored_obs(&a, 0)]);
        let state = f.state().clone();

        let mut resumed =
            Fuser::resume(std::slice::from_ref(&a), state, FuseConfig::default()).unwrap();
        resumed.fuse(&[scored_obs(&a, 0)]);
        assert_eq!(resumed.state().channels[&key("a")].separation.count(), 2.0);
    }

    #[test]
    fn config_accessor_returns_the_configuration_in_force() {
        // A deliberately non-default knob so the accessor is pinned to THIS config.
        let mut cfg = FuseConfig::default();
        cfg.fusion.rrf_eta = 17.0;
        let f = Fuser::new(&[chan("a", None)], cfg).unwrap();
        assert_eq!(f.config(), &cfg);
    }

    #[test]
    fn refresh_coupling_decays_the_refresh_count_multiplicatively() {
        // With decay on at factor 0.5: refreshes go 0 -> 1 -> (1*0.5)+1 = 1.5. The
        // mutants += (2.5) and /= (3.0) both diverge at the second refresh.
        let pair = UnorderedPair::new(key("a"), key("b"));
        let fcfg = FuseConfig {
            decay: DecayConfig {
                enabled: true,
                factor: 0.5,
            },
            ..FuseConfig::default()
        };
        let mut f = fuser(&[chan("a", None), chan("b", None)], fcfg);
        let anchor = anchor_of(40, |id| id as f64);
        f.refresh_coupling(&anchor);
        assert_abs_diff_eq!(f.state.pairs[&pair].refreshes, 1.0, epsilon = 1e-12);
        f.refresh_coupling(&anchor);
        assert_abs_diff_eq!(f.state.pairs[&pair].refreshes, 1.5, epsilon = 1e-12);
    }

    #[test]
    fn declared_reference_seeds_variance_as_sigma_squared() {
        // GoodScore(0.1, 0.5, 6): sigma_ref = (0.5-0.1)/2 = 0.2, so the seeded reference
        // must carry variance 0.04 at count 6. Seeding happens on the channel's first
        // appearance; a ranks-only input triggers it without pushing any read, so the
        // prior is observable exactly as seeded.
        let a = chan("a", Some(GoodScore::new(0.1, 0.5, 6.0)));
        let mut f = fuser(std::slice::from_ref(&a), FuseConfig::default());
        f.fuse(&[ranks_obs(&a, &[1, 2, 3])]);
        let reference = &f.state.channels[&key("a")].reference;
        assert_abs_diff_eq!(reference.mean(), 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(reference.variance(), 0.04, epsilon = 1e-12);
        assert_abs_diff_eq!(reference.count(), 6.0, epsilon = 1e-12);
    }

    // --- 9. determinism ----------------------------------------------------------------

    #[test]
    fn fuse_is_deterministic_for_equal_inputs_and_state() {
        let a = chan("a", Some(GoodScore::new(0.0, 1.0, 5.0)));
        let b = chan("b", None);
        let obs = vec![scored_obs(&a, 0), ranks_obs(&b, &[100, 5, 6, 7])];

        let cfg = CouplingConfig {
            enabled: true,
            min_reliability: 1.0,
            ..CouplingConfig::default()
        };
        let fcfg = FuseConfig {
            coupling: cfg,
            ..FuseConfig::default()
        };
        let mut f1 = fuser(&[a.clone(), b.clone()], fcfg);
        let mut f2 = fuser(&[a.clone(), b.clone()], fcfg);

        let r1 = f1.fuse(&obs);
        let r2 = f2.fuse(&obs);
        assert_eq!(r1.ranking, r2.ranking);
        assert_eq!(r1, r2); // weights, flags, and diagnostics all match
    }
}
