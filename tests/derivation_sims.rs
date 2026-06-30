//! Reproductions of the `ruffle` derivation's own simulations as integration tests.
//!
//! Each test below re-runs a simulation `docs/derivation.md` describes and asserts its
//! QUALITATIVE / DIRECTIONAL finding against the shipped public API, with a fixed-seed
//! RNG and tolerances rather than the derivation's exact decimals. The actual numbers are
//! logged (run with `--nocapture`), so a reader can compare them to the derivation's.
//! Where a simulation does NOT reproduce its finding, the test is marked
//! `#[ignore = "DEFECT: ..."]` and called out, per the defect protocol, rather than
//! tuned until it passes.
//!
//! Simulations covered:
//!   1. §4   — separation tracks informativeness and is scale-free.
//!   2. §4   — absolute goodness covers separation's flat-low/flat-high blind spot.
//!   3. §5.2 — the Berkson collider: the anchor recovers redundancy, the pool does not.
//!   4. §5.3 — anchor redundancy recovery across loadings λ ∈ {0,1,2,3}.
//!   5. §10  — end-to-end nDCG/recall across the three regimes (the adherence test).
//!
//! Everything here touches only the public surface (`discriminate`, `anchor_correlations`,
//! `weighted_rrf`, `Fuser`, …); the private `linalg`/`winsorize_separation` are not used.

use std::collections::{BTreeMap, HashSet};

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use ruffle::components::{anchor_correlations, discriminate, weighted_rrf};
use ruffle::{
    Anchor, ChannelConfig, ChannelId, ChannelInput, CouplingConfig, Direction,
    DiscriminationConfig, FuseConfig, Fuser, MeanVar, RrfConfig, Score, UnorderedPair,
};

/// A caller-side newtype: the only way a bare number becomes a [`Score`] (§7).
struct Sim(f64);
impl Score for Sim {
    fn value(&self) -> f64 {
        self.0
    }
}

/// A standard-normal sample stream over a `ChaCha8Rng` (fixed seed per simulation).
fn standard_normal() -> Normal<f64> {
    Normal::new(0.0, 1.0).expect("0,1 is a valid normal")
}

/// A registered scored channel: higher-is-better, fixed tag, no declared reference.
fn channel(key: &str) -> ChannelConfig {
    ChannelConfig::new(ChannelId::new(key, "sim"), Direction::HigherIsBetter, None)
}

/// An [`RrfConfig`] with a chosen rank constant `η`, for the fusion calls below.
fn rrf(eta: f64) -> RrfConfig {
    let mut c = RrfConfig::default();
    c.rrf_eta = eta;
    c
}

// =====================================================================================
// 1. §4 — separation tracks informativeness and is scale-free.
// =====================================================================================
//
// Proposal (§4): "On a corpus of 2×10^5 documents with a few dozen relevants and a
// channel of tunable signal strength a, D^sep on the channel's own top-400 pool rises
// monotonically with a (4.7→5.0→8.0→10.5→20.4) and is exactly invariant under a ×100
// rescale, while the standard deviation rises with a and scales with the units."
//
// We build that channel: 2e5 docs, ~40 relevants elevated by `a` over a unit-noise bulk,
// take the channel's own top-400 pool, and read `discriminate(...).raw_separation`.

/// Score one tunable channel over a corpus: relevant docs get `a + N(0,1)`, the bulk gets
/// `N(0,1)`. Returns the channel's own top-`pool_k` pool (highest scores first), already a
/// `Scored` observation's worth of `(id, score)`.
fn tunable_channel_pool(
    rng: &mut ChaCha8Rng,
    n_docs: usize,
    n_rel: usize,
    a: f64,
    pool_k: usize,
) -> Vec<(u32, f64)> {
    let normal = standard_normal();
    // Pick `n_rel` distinct relevant ids.
    let mut ids: Vec<u32> = (0..n_docs as u32).collect();
    ids.shuffle(rng);
    let relevant: HashSet<u32> = ids.iter().take(n_rel).copied().collect();

    let mut scored: Vec<(u32, f64)> = (0..n_docs as u32)
        .map(|d| {
            let signal = if relevant.contains(&d) { a } else { 0.0 };
            (d, signal + normal.sample(rng))
        })
        .collect();
    // The channel's own top-k pool (highest first).
    scored.sort_unstable_by(|x, y| y.1.total_cmp(&x.1));
    scored.truncate(pool_k);
    scored
}

/// The raw separation a pool reads against a cold baseline (count 0 still computes it).
fn raw_separation(pool: &[(u32, f64)]) -> f64 {
    let cfg = channel("x");
    let obs = ChannelInput::scored(&cfg, pool.iter().map(|(id, s)| (*id, Sim(*s))).collect());
    discriminate(
        &obs.items,
        &MeanVar::new(),
        &MeanVar::new(),
        &DiscriminationConfig::default(),
    )
    .raw_separation
    .expect("separation defined for the spiked top-400 pool")
}

#[test]
fn sim1_separation_tracks_informativeness_and_is_scale_free() {
    let n_docs = 200_000;
    let n_rel = 40;
    let pool_k = 400;
    // A sweep of signal strengths. The derivation reports a monotone rise; we assert the
    // sequence is strictly increasing and log the values against its 4.7→…→20.4.
    let a_sweep = [2.0_f64, 3.0, 4.0, 6.0, 10.0];

    let mut seps = Vec::new();
    let mut stds = Vec::new();
    for &a in &a_sweep {
        // A fixed per-`a` seed so the run is reproducible and each `a` is comparable.
        let mut rng = ChaCha8Rng::seed_from_u64(0xA11CE + (a as u64));
        let pool = tunable_channel_pool(&mut rng, n_docs, n_rel, a, pool_k);
        let sep = raw_separation(&pool);
        // The pool's own standard deviation, which the derivation notes rises with `a`.
        let mean = pool.iter().map(|(_, s)| *s).sum::<f64>() / pool.len() as f64;
        let var = pool.iter().map(|(_, s)| (*s - mean).powi(2)).sum::<f64>() / pool.len() as f64;
        seps.push(sep);
        stds.push(var.sqrt());
    }

    println!("\n[sim1] §4 separation vs signal strength a (derivation: 4.7→5.0→8.0→10.5→20.4)");
    for (i, &a) in a_sweep.iter().enumerate() {
        println!(
            "  a={a:>5.1}  D^sep={:>8.3}  pool_std={:>7.3}",
            seps[i], stds[i]
        );
    }

    // Monotone in `a`: separation tracks informativeness.
    for w in seps.windows(2) {
        assert!(
            w[1] > w[0],
            "D^sep must rise with signal strength: {:?}",
            seps
        );
    }
    // The pool standard deviation also rises with `a` (it carries the units).
    for w in stds.windows(2) {
        assert!(w[1] > w[0], "pool std should rise with a: {:?}", stds);
    }

    // Scale-freedom: a ×100 rescale of every score leaves D^sep exactly invariant (a ratio
    // of score differences), and a large additive shift leaves it invariant up to f64
    // cancellation.
    let mut rng = ChaCha8Rng::seed_from_u64(0xA11CE + 4);
    let base = tunable_channel_pool(&mut rng, n_docs, n_rel, 4.0, pool_k);
    let d0 = raw_separation(&base);
    let scaled: Vec<(u32, f64)> = base.iter().map(|(id, s)| (*id, s * 100.0)).collect();
    let shifted: Vec<(u32, f64)> = base.iter().map(|(id, s)| (*id, s + 1000.0)).collect();
    let d_scaled = raw_separation(&scaled);
    let d_shifted = raw_separation(&shifted);
    println!("[sim1] scale-free: D^sep base={d0:.6}  ×100={d_scaled:.6}  +1000={d_shifted:.6}");
    assert!(
        (d_scaled - d0).abs() < 1e-9,
        "×100 rescale must leave D^sep exactly invariant: {d0} vs {d_scaled}"
    );
    assert!(
        (d_shifted - d0).abs() < 1e-6,
        "a large additive shift must leave D^sep invariant up to f64 cancellation: {d0} vs {d_shifted}"
    );
}

// =====================================================================================
// 2. §4 — absolute goodness covers separation's flat-low/flat-high blind spot.
// =====================================================================================
//
// Proposal (§4): against a good-score reference (mu_ref, sigma_ref) = (0.35, 0.07),
// compare three pools — flat-low "nothing matches", flat-high "everything matches", and
// "a few genuine hits". Separation cannot tell flat-low from flat-high (both read the
// same value, since they differ only by a constant shift and D^sep is shift-invariant),
// while D^abs reads flat-low clearly NEGATIVE (≈ −1.3) and flat-high clearly POSITIVE
// (≈ +3.0). D^abs = reference.zscore(top_m_average).

#[test]
fn sim2_absolute_goodness_covers_separation_blind_spot() {
    let cfg = channel("c");
    let dcfg = DiscriminationConfig::default();
    // The declared good-score reference, seeded as a prior with a positive pseudo-count so
    // it standardizes from the first query (§8).
    let reference = MeanVar::from_prior(0.35, 0.07 * 0.07, 16.0);

    // A tight, "flat" pool: 60 distinct values evenly spread in [0.20, 0.30], so the bulk
    // is non-degenerate and separation is defined, but there is no standout. Its top-m
    // average sits below the reference's 0.35.
    let flat_low: Vec<(u32, f64)> = (0..60)
        .map(|i| (i, 0.20 + 0.10 * (i as f64) / 59.0))
        .collect();
    // Flat-high is the SAME shape shifted up by a constant, so D^sep is identical by
    // shift-invariance while its top-m average sits well above the reference.
    let flat_high: Vec<(u32, f64)> = flat_low.iter().map(|(id, s)| (*id, s + 0.30)).collect();
    // A few-genuine-hits pool: the same low bulk with 6 items spiked far above it. Now
    // separation is large and the top-m average is lifted by the hits.
    let mut few_hits = flat_low.clone();
    for (j, item) in few_hits.iter_mut().take(6).enumerate() {
        item.1 = 0.75 + 0.01 * j as f64;
    }

    let read = |pool: &[(u32, f64)]| {
        let obs = ChannelInput::scored(&cfg, pool.iter().map(|(id, s)| (*id, Sim(*s))).collect());
        let d = discriminate(&obs.items, &MeanVar::new(), &reference, &dcfg);
        let d_abs = reference
            .zscore(d.top_m_average.expect("top-m average defined"))
            .expect("declared reference standardizes from the first query");
        (d.raw_separation.expect("separation defined"), d_abs)
    };

    let (sep_low, abs_low) = read(&flat_low);
    let (sep_high, abs_high) = read(&flat_high);
    let (sep_hits, abs_hits) = read(&few_hits);

    println!("\n[sim2] §4 absolute goodness vs separation (derivation D^abs: −1.3 vs +3.0)");
    println!("  flat-low   D^sep={sep_low:>7.3}  D^abs={abs_low:>7.3}");
    println!("  flat-high  D^sep={sep_high:>7.3}  D^abs={abs_high:>7.3}");
    println!("  few-hits   D^sep={sep_hits:>7.3}  D^abs={abs_hits:>7.3}");

    // Separation CANNOT distinguish flat-low from flat-high: equal up to f64 rounding.
    assert!(
        (sep_low - sep_high).abs() < 1e-9,
        "separation must be blind to the flat-low/flat-high shift: {sep_low} vs {sep_high}"
    );
    // Absolute goodness DOES: flat-low clearly negative, flat-high clearly positive.
    assert!(
        abs_low < -0.5,
        "flat-low D^abs should be clearly negative: {abs_low}"
    );
    assert!(
        abs_high > 2.0,
        "flat-high D^abs should be clearly positive: {abs_high}"
    );
    assert!(
        abs_low < abs_high,
        "D^abs must order flat-low below flat-high"
    );
    // On the discriminating pool the two statistics agree: high separation AND positive
    // absolute goodness.
    assert!(
        sep_hits > sep_low + 1.0,
        "the few-hits pool should read a clearly higher separation: {sep_hits} vs {sep_low}"
    );
    assert!(
        abs_hits > 0.0,
        "the few-hits pool's D^abs should be positive: {abs_hits}"
    );
}

// =====================================================================================
// Shared two-channel collider generator for §5.2 / §5.3 (sims 3 and 4).
// =====================================================================================
//
// Two channels S_c = a·R + λ·Z + ε_c, with R rare relevance (rate `p`), Z a shared
// nuisance of loading λ, ε_c independent unit noise. The true within-irrelevant
// correlation is λ²/(λ²+1).

/// Generate the two channels' score vectors over `n` documents.
fn collider_scores(
    rng: &mut ChaCha8Rng,
    n: usize,
    a: f64,
    lambda: f64,
    p: f64,
) -> (Vec<f64>, Vec<f64>) {
    let normal = standard_normal();
    let mut sa = Vec::with_capacity(n);
    let mut sb = Vec::with_capacity(n);
    for _ in 0..n {
        let r = if rng.gen_range(0.0_f64..1.0) < p {
            1.0
        } else {
            0.0
        };
        let z = normal.sample(rng);
        sa.push(a * r + lambda * z + normal.sample(rng));
        sb.push(a * r + lambda * z + normal.sample(rng));
    }
    (sa, sb)
}

/// Pearson correlation over a set of paired samples, or `None` when undefined. (We compute
/// the pool correlation OURSELVES here; `ruffle` deliberately never does — that is §5.2's
/// point.)
fn pearson_over(indices: impl Iterator<Item = usize>, xs: &[f64], ys: &[f64]) -> Option<f64> {
    let idx: Vec<usize> = indices.collect();
    let n = idx.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mx = idx.iter().map(|&i| xs[i]).sum::<f64>() / nf;
    let my = idx.iter().map(|&i| ys[i]).sum::<f64>() / nf;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for &i in &idx {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        return None;
    }
    Some(cov / (vx.sqrt() * vy.sqrt()))
}

/// The union of each channel's top-`k` indices (the live fusion pool).
fn union_top_k(sa: &[f64], sb: &[f64], k: usize) -> HashSet<usize> {
    let top = |s: &[f64]| -> HashSet<usize> {
        let mut idx: Vec<usize> = (0..s.len()).collect();
        idx.sort_unstable_by(|&i, &j| s[j].total_cmp(&s[i]));
        idx.into_iter().take(k).collect()
    };
    let mut u = top(sa);
    u.extend(top(sb));
    u
}

/// Recover the anchor's both-scored correlation for the single pair via the PUBLIC
/// `anchor_correlations` on a full-scored, unselected anchor (exactly what `ruffle` does).
fn anchor_pair_correlation(sa: &[f64], sb: &[f64]) -> f64 {
    let ca = channel("a");
    let cb = channel("b");
    let cands: Vec<usize> = (0..sa.len()).collect();
    let anchor = Anchor::build(&cands, &[&ca, &cb], |id, k| {
        let v = if k == "a" { sa[*id] } else { sb[*id] };
        Some(Sim(v))
    });
    let corr = anchor_correlations(&anchor, &CouplingConfig::default());
    corr.get(&UnorderedPair::new("a".to_string(), "b".to_string()))
        .expect("the pair clears the default min_overlap on a full anchor")
        .correlation
}

// =====================================================================================
// 3. §5.2 — the Berkson collider.
// =====================================================================================
//
// Proposal table (N=2e5, p=1e-3, a=3, k=1000):
//   λ : within-irrelevant (random sample) | theory λ²/(λ²+1) | full corr on the pool
//   0 :  0.00                              | 0.00            | −0.63
//   1 :  0.50                              | 0.50            | −0.52
//   2 :  0.80                              | 0.80            | −0.18
//   3 :  0.90                              | 0.90            | +0.16
//
// We assert: the anchor (unselected) recovers λ²/(λ²+1); the union-of-top-k pool
// correlation is strongly biased downward (negative at low λ) and does NOT track the
// theory. This is WHY `ruffle` reads coupling off the anchor, not the pool.

#[test]
fn sim3_berkson_collider_pool_is_biased_anchor_is_not() {
    let n = 200_000;
    let a = 3.0;
    let p = 1e-3;
    let k = 1000;
    let lambdas = [0.0_f64, 1.0, 2.0, 3.0];

    println!("\n[sim3] §5.2 Berkson collider (N={n}, p={p}, a={a}, k={k})");
    println!("  λ    theory    anchor(unselected)   pool(union top-k)");
    for &lambda in &lambdas {
        let mut rng = ChaCha8Rng::seed_from_u64(0xB3450 + (lambda as u64));
        let (sa, sb) = collider_scores(&mut rng, n, a, lambda, p);
        let theory = lambda * lambda / (lambda * lambda + 1.0);

        let anchor = anchor_pair_correlation(&sa, &sb);
        let pool_idx = union_top_k(&sa, &sb, k);
        let pool = pearson_over(pool_idx.into_iter(), &sa, &sb).expect("pool correlation defined");

        println!("  {lambda:.0}    {theory:>5.2}     {anchor:>7.3}            {pool:>7.3}");

        // The anchor recovers the true within-irrelevant redundancy to ~2 digits (the
        // rare relevance only nudges it, since the sample is bulk-dominated).
        assert!(
            (anchor - theory).abs() < 0.05,
            "anchor must recover λ²/(λ²+1)={theory:.2}, got {anchor:.3} (λ={lambda})"
        );
        // The pool correlation is strongly biased DOWN by the collider: far below the
        // truth, and negative at low loadings.
        assert!(
            pool < theory - 0.3,
            "the union-of-top-k pool correlation must be biased well below the truth: \
             pool={pool:.3} vs theory={theory:.2} (λ={lambda})"
        );
        if lambda <= 1.0 {
            assert!(
                pool < 0.0,
                "at low λ the selection drives the pool correlation negative: {pool:.3} (λ={lambda})"
            );
        }
    }
}

// =====================================================================================
// 4. §5.3 — anchor redundancy recovery across loadings.
// =====================================================================================
//
// Proposal (§5.2 "random sample" column / §5.3): the anchor recovers the injected
// redundancy {0.0, 0.5, 0.8, 0.9} for λ ∈ {0,1,2,3} to ~2 digits. A smaller, faster
// unselected sample suffices for the anchor to converge.

#[test]
fn sim4_anchor_recovers_redundancy_across_loadings() {
    let n = 20_000;
    let a = 3.0;
    let p = 1e-3;
    let expected = [0.0_f64, 0.5, 0.8, 0.9];
    let lambdas = [0.0_f64, 1.0, 2.0, 3.0];

    println!("\n[sim4] §5.3 anchor redundancy recovery (target {{0.0, 0.5, 0.8, 0.9}})");
    for (i, &lambda) in lambdas.iter().enumerate() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5EED0 + (lambda as u64));
        let (sa, sb) = collider_scores(&mut rng, n, a, lambda, p);
        let anchor = anchor_pair_correlation(&sa, &sb);
        println!(
            "  λ={lambda:.0}  expected≈{:.2}  anchor={anchor:.3}",
            expected[i]
        );
        assert!(
            (anchor - expected[i]).abs() < 0.05,
            "anchor redundancy for λ={lambda} should be ≈{:.2}, got {anchor:.3}",
            expected[i]
        );
    }
}

// =====================================================================================
// 5. §10 — end-to-end nDCG / recall across the three regimes (the adherence test).
// =====================================================================================
//
// A synthetic corpus with KNOWN relevance labels and four channels, one redundant pair
// (channels 0 and 1 share a nuisance Z of loading λ). We compare three fusions over many
// seeded queries:
//   - unweighted RRF: weighted_rrf with all weights 1.
//   - ruffle-weighted: Fuser::fuse, after a burn-in warms the baselines and a
//     refresh_coupling on an UNSELECTED anchor discounts the redundant pair.
//   - oracle-weighted: weighted_rrf with weights ∝ each channel's TRUE per-query gain.
//
// Three regimes (derivation §10):
//   (a) per-query VARYING informativeness → ruffle clearly ABOVE unweighted, near oracle
//       (derivation nDCG@10 0.61→0.90, oracle 0.91).
//   (b) near-EQUAL channels → ruffle ≈ ties unweighted (within a small margin).
//   (c) purely STATIC-differing → ruffle does NOT beat unweighted and may UNDERPERFORM
//       (derivation 0.52→0.35), the documented §9/§10 limitation. Reproduced HONESTLY.

const N_CH: usize = 4;

#[derive(Clone, Copy, PartialEq)]
enum Regime {
    Varying,
    Equal,
    Static,
}

/// Per-query channel gains under the regime. The redundant pair is always channels 0,1.
fn gains_for(regime: Regime, rng: &mut ChaCha8Rng) -> [f64; N_CH] {
    match regime {
        // A clearly-best channel each query, but WHICH channel varies (a permutation of a
        // fixed gain vector). ruffle's within-channel standardization can see this.
        Regime::Varying => {
            let base = [2.2_f64, 1.3, 0.7, 0.3];
            let mut idx = [0usize, 1, 2, 3];
            idx.shuffle(rng);
            let mut g = [0.0; N_CH];
            for c in 0..N_CH {
                g[c] = base[idx[c]];
            }
            g
        }
        // All channels equally (and decently) informative every query: nothing to
        // differentiate, so the per-query diagonal should read neutral and ruffle should
        // not move off plain RRF.
        Regime::Equal => [1.8; N_CH],
        // Channels differ in a FIXED way with no per-query variation. The redundant pair
        // (0,1) differs statically (2.2 vs 0.4), so the symmetric redundancy discount
        // transfers weight off the sharp channel — the §9 static-differing harm.
        Regime::Static => [2.2, 0.4, 1.4, 0.8],
    }
}

/// One channel's unit-variance noise for a document. The redundant pair — channels 0 and
/// channel 1 — draws `√ρ · Z_shared + √(1−ρ) · ε_c`; the independent channels draw `ε_c`.
/// Every channel's total noise variance is exactly `1`, so the redundant pair is NOT a
/// worse ranker — it only DOUBLE-COUNTS. This isolates the per-query discrimination
/// (diagonal) as the thing the three regimes vary, with the redundancy a constant,
/// separate axis the coupling discount corrects. The pair's noise correlation is `ρ` (§5.1).
fn channel_noise(
    rng: &mut ChaCha8Rng,
    normal: &Normal<f64>,
    c: usize,
    rho: f64,
    z_shared: f64,
) -> f64 {
    if c == 0 || c == 1 {
        rho.sqrt() * z_shared + (1.0 - rho).sqrt() * normal.sample(rng)
    } else {
        normal.sample(rng)
    }
}

/// Build one query: four channels each scoring `n_docs` docs and returning their own
/// top-`pool_k` pool. Relevant docs get `gain + noise`, the bulk gets `noise`, with the
/// unit-variance redundant-pair noise above. Returns the observations and the relevant ids.
fn build_query(
    rng: &mut ChaCha8Rng,
    cfgs: &[ChannelConfig],
    gains: [f64; N_CH],
    rho: f64,
    n_docs: usize,
    n_rel: usize,
    pool_k: usize,
) -> (Vec<ChannelInput<u32>>, HashSet<u32>) {
    let normal = standard_normal();
    let mut ids: Vec<u32> = (0..n_docs as u32).collect();
    ids.shuffle(rng);
    let relevant: HashSet<u32> = ids.iter().take(n_rel).copied().collect();
    let z: Vec<f64> = (0..n_docs).map(|_| normal.sample(rng)).collect();

    let mut obs = Vec::with_capacity(N_CH);
    for c in 0..N_CH {
        let mut scored: Vec<(u32, f64)> = (0..n_docs)
            .map(|d| {
                let signal = if relevant.contains(&(d as u32)) {
                    gains[c]
                } else {
                    0.0
                };
                (d as u32, signal + channel_noise(rng, &normal, c, rho, z[d]))
            })
            .collect();
        scored.sort_unstable_by(|x, y| y.1.total_cmp(&x.1));
        scored.truncate(pool_k);
        obs.push(ChannelInput::scored(
            &cfgs[c],
            scored.into_iter().map(|(id, s)| (id, Sim(s))).collect(),
        ));
    }
    (obs, relevant)
}

/// Build a full-scored UNSELECTED anchor that carries only the nuisance structure: the
/// same unit-variance noise as the live channels, no relevance signal. The redundant pair
/// (0,1) correlates at `ρ`, the others at `0`. This is what `refresh_coupling` reads the
/// redundancy from (§5.3).
fn build_anchor(rng: &mut ChaCha8Rng, cfgs: &[ChannelConfig], rho: f64, n: usize) -> Anchor {
    let normal = standard_normal();
    let z: Vec<f64> = (0..n).map(|_| normal.sample(rng)).collect();
    let mut mat = [(); N_CH].map(|_| Vec::<f64>::with_capacity(n));
    for (c, row) in mat.iter_mut().enumerate() {
        for &zd in &z {
            row.push(channel_noise(rng, &normal, c, rho, zd));
        }
    }
    let cands: Vec<u32> = (0..n as u32).collect();
    let refs: Vec<&ChannelConfig> = cfgs.iter().collect();
    Anchor::build(&cands, &refs, |id, key| {
        let c = channel_index(key);
        Some(Sim(mat[c][*id as usize]))
    })
}

/// Channel index from its key string `"ch0".."ch3"`.
fn channel_index(key: &str) -> usize {
    key[2..].parse::<usize>().expect("channel key is ch<idx>")
}

/// Binary-relevance DCG@k of a ranking (best first).
fn dcg_at_k(ranking: &[(u32, f64)], relevant: &HashSet<u32>, k: usize) -> f64 {
    ranking
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, (id, _))| relevant.contains(id))
        .map(|(i, _)| 1.0 / ((i + 2) as f64).log2())
        .sum()
}

/// nDCG@k against the relevant set (binary gains).
fn ndcg_at_k(ranking: &[(u32, f64)], relevant: &HashSet<u32>, k: usize) -> f64 {
    let dcg = dcg_at_k(ranking, relevant, k);
    let ideal_hits = relevant.len().min(k);
    let idcg: f64 = (0..ideal_hits).map(|i| 1.0 / ((i + 2) as f64).log2()).sum();
    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

/// recall@k against the relevant set.
fn recall_at_k(ranking: &[(u32, f64)], relevant: &HashSet<u32>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = ranking
        .iter()
        .take(k)
        .filter(|(id, _)| relevant.contains(id))
        .count();
    hits as f64 / relevant.len() as f64
}

/// The averaged metrics for one regime: (unweighted, ruffle, oracle) for nDCG@10 and
/// recall@100.
struct RegimeResult {
    unw_ndcg: f64,
    ruffle_ndcg: f64,
    oracle_ndcg: f64,
    unw_recall: f64,
    ruffle_recall: f64,
    oracle_recall: f64,
}

/// Run one regime end-to-end: warm ruffle's baselines over a burn-in, refresh coupling on
/// an unselected anchor, then evaluate the three fusions over the eval queries.
fn run_regime(regime: Regime, seed: u64) -> RegimeResult {
    let n_docs = 1500;
    let n_rel = 30;
    let pool_k = 300;
    let rho = 0.5; // redundant-pair noise correlation (equal-variance, so a pure redundancy)
    let eta = FuseConfig::default().fusion.rrf_eta;
    let burn_in = 150;
    let eval = 300;
    let anchor_n = 2000;

    let cfgs: Vec<ChannelConfig> = (0..N_CH).map(|c| channel(&format!("ch{c}"))).collect();

    // ruffle runs with coupling ENABLED so the refreshed redundancy is actually applied.
    let mut fuse_cfg = FuseConfig::default();
    fuse_cfg.coupling.enabled = true;
    let mut fuser = Fuser::new(&cfgs, fuse_cfg).unwrap();

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Burn-in: warm the per-channel separation baselines and learned references.
    for _ in 0..burn_in {
        let gains = gains_for(regime, &mut rng);
        let (obs, _rel) = build_query(&mut rng, &cfgs, gains, rho, n_docs, n_rel, pool_k);
        fuser.fuse(&obs);
    }

    // Refresh the coupling off-diagonal on an unselected anchor: the redundant pair is now
    // discounted (§5.3).
    let anchor = build_anchor(&mut rng, &cfgs, rho, anchor_n);
    fuser.refresh_coupling(&anchor);

    let (mut un, mut rn, mut on) = (0.0, 0.0, 0.0);
    let (mut ur, mut rr, mut or_) = (0.0, 0.0, 0.0);
    for _ in 0..eval {
        let gains = gains_for(regime, &mut rng);
        let (obs, relevant) = build_query(&mut rng, &cfgs, gains, rho, n_docs, n_rel, pool_k);

        // Unweighted RRF: empty map => every channel defaults to weight 1.
        let r_unw = weighted_rrf(&obs, &BTreeMap::new(), &rrf(eta));
        // Oracle: weights ∝ true per-query gain, normalized to sum N.
        let gsum: f64 = gains.iter().sum();
        let oracle_w: BTreeMap<String, f64> = (0..N_CH)
            .map(|c| (cfgs[c].id.key.clone(), N_CH as f64 * gains[c] / gsum))
            .collect();
        let r_oracle = weighted_rrf(&obs, &oracle_w, &rrf(eta));
        // ruffle: stateful fuse (also keeps learning during eval).
        let r_ruffle = fuser.fuse(&obs);

        un += ndcg_at_k(&r_unw, &relevant, 10);
        rn += ndcg_at_k(&r_ruffle.ranking, &relevant, 10);
        on += ndcg_at_k(&r_oracle, &relevant, 10);
        ur += recall_at_k(&r_unw, &relevant, 100);
        rr += recall_at_k(&r_ruffle.ranking, &relevant, 100);
        or_ += recall_at_k(&r_oracle, &relevant, 100);
    }
    let m = eval as f64;
    RegimeResult {
        unw_ndcg: un / m,
        ruffle_ndcg: rn / m,
        oracle_ndcg: on / m,
        unw_recall: ur / m,
        ruffle_recall: rr / m,
        oracle_recall: or_ / m,
    }
}

fn log_regime(name: &str, r: &RegimeResult) {
    println!("\n[sim5] regime: {name}");
    println!(
        "  nDCG@10   unweighted={:.3}  ruffle={:.3}  oracle={:.3}",
        r.unw_ndcg, r.ruffle_ndcg, r.oracle_ndcg
    );
    println!(
        "  recall@100 unweighted={:.3}  ruffle={:.3}  oracle={:.3}",
        r.unw_recall, r.ruffle_recall, r.oracle_recall
    );
}

#[test]
fn sim5a_varying_informativeness_ruffle_beats_rrf_and_approaches_oracle() {
    let r = run_regime(Regime::Varying, 0x5151_000A);
    log_regime(
        "(a) per-query VARYING informativeness (derivation 0.61→0.90, oracle 0.91)",
        &r,
    );

    // ruffle clearly above unweighted RRF.
    assert!(
        r.ruffle_ndcg > r.unw_ndcg + 0.05,
        "ruffle should clearly beat unweighted RRF when informativeness varies per query: \
         ruffle={:.3} vs unweighted={:.3}",
        r.ruffle_ndcg,
        r.unw_ndcg
    );
    // ruffle approaches the oracle (captures most of the achievable gap). The oracle here
    // is gain-proportional only, so it is NOT redundancy-aware and not a strict ceiling;
    // ruffle can even edge it out via the coupling discount. We only require ruffle to land
    // near it (within a margin on the low side).
    assert!(
        r.ruffle_ndcg >= r.oracle_ndcg - 0.10,
        "ruffle should approach the oracle: ruffle={:.3} vs oracle={:.3}",
        r.ruffle_ndcg,
        r.oracle_ndcg
    );
}

#[test]
fn sim5b_near_equal_channels_ruffle_ties_rrf() {
    let r = run_regime(Regime::Equal, 0x5151_000B);
    log_regime("(b) near-EQUAL channels (derivation: ≈ ties plain RRF)", &r);

    // "Roughly ties": the per-query diagonal has nothing to exploit and the coupling
    // discount on the (equally good) redundant pair gives no net benefit, so ruffle lands
    // within a small margin of plain RRF — typically a touch below it, since with neither
    // per-query variation nor a genuinely-worse channel to drop, the weighting only adds
    // noise (§9). We require it not to drift far in either direction.
    let margin = 0.06;
    assert!(
        (r.ruffle_ndcg - r.unw_ndcg).abs() < margin,
        "ruffle should roughly tie unweighted RRF when channels are near-equal \
         (within {margin}): ruffle={:.3} vs unweighted={:.3}",
        r.ruffle_ndcg,
        r.unw_ndcg
    );
}

#[test]
fn sim5c_static_differing_ruffle_does_not_beat_rrf() {
    let r = run_regime(Regime::Static, 0x5151_000C);
    log_regime(
        "(c) purely STATIC-differing (derivation 0.52→0.35: ruffle underperforms)",
        &r,
    );
    let gap = r.unw_ndcg - r.ruffle_ndcg;
    println!("  honest gap (unweighted − ruffle) = {gap:+.3}  (positive ⇒ ruffle underperforms)");

    // The documented §9/§10 limitation: with no per-query signal to exploit, ruffle must
    // NOT improve on unweighted RRF. This is the portable, primary claim — assert it does
    // not beat unweighted by more than a small noise margin.
    let margin = 0.02;
    assert!(
        r.ruffle_ndcg <= r.unw_ndcg + margin,
        "in the purely static-differing regime ruffle must NOT improve on unweighted RRF \
         (the §9 limitation): ruffle={:.3} vs unweighted={:.3} (gap {gap:+.3})",
        r.ruffle_ndcg,
        r.unw_ndcg
    );
    // The derivation goes further (0.52→0.35): ruffle actively LOSES here, because the
    // per-query diagonal's noise plus the symmetric redundancy discount on the
    // statically-sharp channel move weight the wrong way. We reproduce that loss honestly:
    // ruffle underperforms unweighted by a clear margin. (Should a future §9 do-no-harm
    // allocation remove the loss, this assertion is meant to flag it loudly.)
    assert!(
        r.ruffle_ndcg < r.unw_ndcg - 0.03,
        "the static-differing regime should reproduce the §9/§10 LOSS (ruffle below RRF): \
         ruffle={:.3} vs unweighted={:.3} (gap {gap:+.3})",
        r.ruffle_ndcg,
        r.unw_ndcg
    );
    // The oracle, which sees the static gains, still does best — confirming the signal is
    // there to exploit and ruffle simply cannot read it within-channel.
    assert!(
        r.oracle_ndcg > r.unw_ndcg,
        "the oracle should beat unweighted RRF in the static regime: oracle={:.3} vs unweighted={:.3}",
        r.oracle_ndcg,
        r.unw_ndcg
    );
}
