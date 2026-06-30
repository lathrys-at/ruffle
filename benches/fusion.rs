//! Benchmarks for the per-query hot path and the state operations around it.
//!
//! The shapes mirror the design target (§11): a handful of channels, pools of a few
//! hundred to a thousand candidates per query, warm baselines. Fusion sits between a
//! retrieval fan-out and a reranker, so the number that matters is that a full weighted
//! fuse stays comfortably in the microsecond range next to the milliseconds the
//! retrieval itself costs.

// The crate denies missing_docs; criterion's harness macros expand to undocumented
// items, which is fine in a bench target.
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruffle::components::weighted_rrf;
use ruffle::{
    ChannelConfig, ChannelId, ChannelInput, Direction, FuseConfig, Fuser, GoodScore, MergePolicy,
    RuffleState, Score,
};
use std::hint::black_box;

struct Sim(f64);
impl Score for Sim {
    fn value(&self) -> f64 {
        self.0
    }
}

const CHANNELS: usize = 4;

fn configs() -> Vec<ChannelConfig> {
    (0..CHANNELS)
        .map(|c| {
            ChannelConfig::new(
                ChannelId::new(format!("channel-{c}"), "bench-v1"),
                Direction::HigherIsBetter,
                Some(GoodScore::new(0.30, 0.44, 8.0)),
            )
        })
        .collect()
}

/// One query's inputs: each channel scores `k` candidates from a shared universe, with
/// a small elevated top so the pools are realistically shaped (bulk + standouts).
fn query_inputs(cfgs: &[ChannelConfig], k: usize, seed: u64) -> Vec<ChannelInput<u64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    cfgs.iter()
        .map(|cfg| {
            let mut items: Vec<(u64, Sim)> = (0..k)
                .map(|_| {
                    let id = rng.gen_range(0..(4 * k as u64));
                    (id, Sim(rng.gen_range(0.05..0.35)))
                })
                .collect();
            for slot in items.iter_mut().take(k / 50) {
                slot.1 = Sim(rng.gen_range(0.40..0.55)); // the elevated top
            }
            // The fusion precondition: distinct ids within one channel.
            items.sort_by_key(|(id, _)| *id);
            items.dedup_by_key(|(id, _)| *id);
            ChannelInput::scored(cfg, items)
        })
        .collect()
}

/// A fuser whose baselines have absorbed enough queries to leave cold start.
fn warm_fuser(cfgs: &[ChannelConfig], k: usize) -> Fuser {
    let mut fuser = Fuser::new(cfgs, FuseConfig::default()).expect("valid registrations");
    for seed in 0..16 {
        let obs = query_inputs(cfgs, k, seed);
        fuser.fuse(&obs);
    }
    fuser
}

fn bench_fuse(c: &mut Criterion) {
    let cfgs = configs();
    let mut group = c.benchmark_group("fuse");
    for k in [100usize, 1000] {
        let mut fuser = warm_fuser(&cfgs, k);
        let obs = query_inputs(&cfgs, k, 999);
        group.throughput(Throughput::Elements((CHANNELS * k) as u64));
        group.bench_with_input(BenchmarkId::new("4ch_stateful", k), &k, |b, _| {
            b.iter(|| black_box(fuser.fuse(black_box(&obs))));
        });
    }
    group.finish();
}

fn bench_weighted_rrf(c: &mut Criterion) {
    let cfgs = configs();
    let mut group = c.benchmark_group("weighted_rrf");
    for k in [100usize, 1000] {
        let obs = query_inputs(&cfgs, k, 7);
        let weights = std::collections::BTreeMap::new(); // neutral weights
        let rrf = FuseConfig::default().fusion;
        group.throughput(Throughput::Elements((CHANNELS * k) as u64));
        group.bench_with_input(BenchmarkId::new("4ch", k), &k, |b, _| {
            b.iter(|| black_box(weighted_rrf(black_box(&obs), &weights, &rrf)));
        });
    }
    group.finish();
}

fn bench_state_merge(c: &mut Criterion) {
    let cfgs = configs();
    let a = warm_fuser(&cfgs, 500).state().clone();
    let b = warm_fuser(&cfgs, 500).state().clone();
    c.bench_function("state_merge_2x4ch", |bench| {
        bench.iter(|| {
            black_box(
                RuffleState::merge(&[black_box(&a), black_box(&b)], MergePolicy::Strict)
                    .expect("compatible states"),
            )
        });
    });
}

criterion_group!(benches, bench_fuse, bench_weighted_rrf, bench_state_merge);
criterion_main!(benches);
