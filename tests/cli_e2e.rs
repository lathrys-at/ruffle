//! CLI process tests for the `ruffle` binary (§8, §11).
//!
//! Each test builds real `RuffleState`s in-process via the library, serializes them to a
//! tempdir, then drives the BUILT binary (`CARGO_BIN_EXE_ruffle`) with
//! `std::process::Command`, asserting on exit status, stdout, stderr, and the files the
//! process did or did not write. The point is the binary behaving as a unit: it must
//! merge what the library merges, refuse without writing on a corruption, warn without
//! dropping on an asymmetric input, and fail cleanly on bad I/O.
//!
//! Gated on the `cli` feature, so a no-feature `cargo test` compiles this to nothing
//! (the bin, and its `CARGO_BIN_EXE` path, exist only under `--features cli`).
#![cfg(feature = "cli")]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruffle::{
    ChannelConfig, ChannelId, ChannelInput, Direction, FuseConfig, Fuser, MergePolicy, RuffleState,
    Score,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

// --- the caller-side score newtype: the only way a bare number becomes a Score (§7) ---

struct Sim(f64);
impl Score for Sim {
    fn value(&self) -> f64 {
        self.0
    }
}

fn key(s: &str) -> String {
    s.to_string()
}

fn cfg(name: &str, tag: &str) -> ChannelConfig {
    ChannelConfig::new(ChannelId::new(name, tag), Direction::HigherIsBetter, None)
}

/// A spiked pool with a well-defined separation statistic (§4).
fn spiked(rng: &mut ChaCha8Rng, base: u32) -> Vec<(u32, Sim)> {
    let mut v: Vec<(u32, Sim)> = (0..30)
        .map(|i| (base + i, Sim(rng.r#gen::<f64>())))
        .collect();
    v.push((base + 1_000, Sim(5.0)));
    v.push((base + 1_001, Sim(5.5)));
    v
}

/// Grow a real `RuffleState` by fusing `n` seeded queries over `cfgs`.
fn grow(cfgs: &[ChannelConfig], seed: u64, n: usize) -> RuffleState {
    let mut f = Fuser::new(cfgs, FuseConfig::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..n {
        let obs: Vec<ChannelInput<u32>> = cfgs
            .iter()
            .enumerate()
            .map(|(i, c)| ChannelInput::scored(c, spiked(&mut rng, (i as u32) * 10_000)))
            .collect();
        f.fuse(&obs);
    }
    f.state().clone()
}

fn dump(path: &Path, s: &RuffleState) {
    std::fs::write(path, serde_json::to_string(s).unwrap()).unwrap();
}

fn slurp(path: &Path) -> RuffleState {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn ruffle() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ruffle"))
}

fn stdout_of(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr_of(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

// --- 1. reconcile: exit 0, output == library merge, report carries divergence+partition

#[test]
fn reconcile_writes_the_library_merge_and_reports() {
    let cfgs = [cfg("a", "ta"), cfg("b", "tb")];
    let a = grow(&cfgs, 1, 12);
    let b = grow(&cfgs, 2, 9);

    let dir = tempfile::tempdir().unwrap();
    let (pa, pb) = (dir.path().join("a.state"), dir.path().join("b.state"));
    let merged_path = dir.path().join("merged.state");
    dump(&pa, &a);
    dump(&pb, &b);

    let out = ruffle()
        .args(["reconcile"])
        .arg(&pa)
        .arg(&pb)
        .arg("-o")
        .arg(&merged_path)
        .arg("--report")
        .output()
        .unwrap();

    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    // The written state equals the library merge of the same two inputs, byte-for-byte
    // through the parse (float_roundtrip makes the reload bit-exact).
    let written = slurp(&merged_path);
    let (expected, _) = RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap();
    assert_eq!(
        written, expected,
        "the CLI writes exactly the library merge"
    );
    assert!((written.channels[&key("a")].separation.count() - 21.0).abs() < 1e-9); // 12 + 9

    // The report mentions the advisory divergence and the merged/carried partition.
    let report = stdout_of(&out);
    assert!(report.contains("divergence"), "report: {report}");
    assert!(
        report.contains("max:"),
        "report names the gated number: {report}"
    );
    assert!(
        report.contains("merged"),
        "report names the merged set: {report}"
    );
    assert!(
        report.contains("carried"),
        "report names the carried set: {report}"
    );
}

// --- 2. tag mismatch on a shared channel: refuse, name it, write nothing --------------

#[test]
fn tag_mismatch_refuses_without_writing_output() {
    // Same key "x", different tags: a model swap under a kept name (§8).
    let a = grow(&[cfg("x", "model-1")], 10, 5);
    let b = grow(&[cfg("x", "model-2")], 11, 5);

    let dir = tempfile::tempdir().unwrap();
    let (pa, pb) = (dir.path().join("a.state"), dir.path().join("b.state"));
    let merged_path = dir.path().join("merged.state");
    dump(&pa, &a);
    dump(&pb, &b);

    let out = ruffle()
        .args(["reconcile"])
        .arg(&pa)
        .arg(&pb)
        .arg("-o")
        .arg(&merged_path)
        .output()
        .unwrap();

    assert!(!out.status.success(), "a tag swap must exit non-zero");
    assert!(
        !merged_path.exists(),
        "a refused merge must not write output"
    );
    let stderr = stderr_of(&out);
    assert!(
        stderr.contains("tag mismatch"),
        "stderr explains the refusal: {stderr}"
    );
    assert!(
        stderr.contains('x'),
        "stderr names the corrupt channel: {stderr}"
    );
}

// --- 3. asymmetric key: warn (non-fatal), carry the channel through the union ---------

#[test]
fn asymmetric_key_warns_but_carries_the_channel() {
    // a knows channels x and y; b knows only x (same tag for x, so x merges cleanly).
    let a = grow(&[cfg("x", "tx"), cfg("y", "ty")], 20, 6);
    let b = grow(&[cfg("x", "tx")], 21, 4);

    let dir = tempfile::tempdir().unwrap();
    let (pa, pb) = (dir.path().join("a.state"), dir.path().join("b.state"));
    let merged_path = dir.path().join("merged.state");
    dump(&pa, &a);
    dump(&pb, &b);

    let out = ruffle()
        .args(["reconcile"])
        .arg(&pa)
        .arg(&pb)
        .arg("-o")
        .arg(&merged_path)
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "an asymmetric input set is a warning, not a failure"
    );
    let stderr = stderr_of(&out);
    assert!(stderr.contains("warning"), "stderr warns: {stderr}");
    assert!(
        stderr.contains('y'),
        "stderr names the asymmetric channel: {stderr}"
    );

    // y is present in only one input, but it is carried through, not dropped.
    let written = slurp(&merged_path);
    assert!(
        written.channels.contains_key(&key("y")),
        "the asymmetric channel survives"
    );
    assert!(written.channels.contains_key(&key("x")));
}

// --- 4. rekey: rename a channel, sanity-check divergence in the report ----------------

#[test]
fn rekey_renames_and_reports_sanity_divergence() {
    let s = grow(&[cfg("old", "to"), cfg("keep", "tk")], 30, 8);

    let dir = tempfile::tempdir().unwrap();
    let inp = dir.path().join("in.state");
    let outp = dir.path().join("out.state");
    dump(&inp, &s);

    let out = ruffle()
        .args(["rekey"])
        .arg(&inp)
        .args(["--from", "old", "--to", "new"])
        .arg("-o")
        .arg(&outp)
        .arg("--report")
        .output()
        .unwrap();

    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let written = slurp(&outp);
    assert!(
        written.channels.contains_key(&key("new")),
        "the renamed channel is present"
    );
    assert!(
        !written.channels.contains_key(&key("old")),
        "the old key is gone"
    );
    assert!(
        written.channels.contains_key(&key("keep")),
        "untouched channels remain"
    );
    // The renamed channel kept its history.
    assert_eq!(
        written.channels[&key("new")].separation.count(),
        s.channels[&key("old")].separation.count()
    );

    let report = stdout_of(&out);
    assert!(report.contains("sanity check"), "report: {report}");
    assert!(
        report.contains("max:"),
        "report prints the divergence: {report}"
    );
    // `keep` is untouched, so its sanity divergence reads zero.
    assert!(
        report.contains("keep: 0.000000"),
        "untouched channel reads zero: {report}"
    );
}

// --- 5. bad input: a missing or malformed file fails cleanly, writing nothing ---------

#[test]
fn unreadable_input_fails_cleanly_without_output() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("does-not-exist.state");
    let outp = dir.path().join("out.state");

    let out = ruffle()
        .args(["reconcile"])
        .arg(&missing)
        .arg("-o")
        .arg(&outp)
        .output()
        .unwrap();

    assert!(!out.status.success(), "a missing input must exit non-zero");
    assert!(!outp.exists(), "no output on a read failure");
    let stderr = stderr_of(&out);
    assert!(
        stderr.contains("cannot read"),
        "stderr explains the read failure: {stderr}"
    );
}

#[test]
fn unparseable_input_fails_cleanly_without_output() {
    let dir = tempfile::tempdir().unwrap();
    let bad: PathBuf = dir.path().join("bad.state");
    std::fs::write(&bad, "{ not valid ruffle state ]").unwrap();
    let outp = dir.path().join("out.state");

    let out = ruffle()
        .args(["reconcile"])
        .arg(&bad)
        .arg("-o")
        .arg(&outp)
        .output()
        .unwrap();

    assert!(
        !out.status.success(),
        "a malformed input must exit non-zero"
    );
    assert!(!outp.exists(), "no output on a parse failure");
    let stderr = stderr_of(&out);
    assert!(
        stderr.contains("cannot parse"),
        "stderr explains the parse failure: {stderr}"
    );
}
