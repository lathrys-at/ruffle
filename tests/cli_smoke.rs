//! End-to-end smoke test for the `ruffle` binary: spawn the built process on temp
//! files and check exit codes and refusal behaviour. Gated on the `cli` feature, so a
//! no-feature `cargo test` compiles this to nothing (the bin, and its CARGO_BIN_EXE
//! path, exist only under `--features cli`).
#![cfg(feature = "cli")]

use ruffle::{BaselineMode, Direction};
use ruffle::{ChannelSummary, RuffleState, StatFingerprint};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

fn fingerprint(dirs: &[(&str, Direction)]) -> StatFingerprint {
    let mut m = BTreeMap::new();
    for (k, d) in dirs {
        m.insert(k.to_string(), *d);
    }
    StatFingerprint::new(BaselineMode::ZScore, m)
}

fn state_with(tag: &str, sep: &[f64]) -> RuffleState {
    let mut s = RuffleState::new(fingerprint(&[("x", Direction::HigherIsBetter)]));
    let mut c = ChannelSummary::new(tag.to_string());
    for &v in sep {
        c.separation.push(v);
    }
    s.channels.insert("x".to_string(), c);
    s
}

fn dump(path: &Path, s: &RuffleState) {
    std::fs::write(path, serde_json::to_string(s).unwrap()).unwrap();
}

#[test]
fn reconcile_succeeds_then_refuses_on_tag_swap() {
    let bin = env!("CARGO_BIN_EXE_ruffle");
    let dir = tempfile::tempdir().unwrap();

    // Compatible: same tag. Merge succeeds, exit 0, output exists and parses.
    let a = dir.path().join("a.state");
    let b = dir.path().join("b.state");
    let merged = dir.path().join("merged.state");
    dump(&a, &state_with("model-1", &[1.0, 2.0]));
    dump(&b, &state_with("model-1", &[3.0, 4.0]));

    let ok = Command::new(bin)
        .args(["reconcile"])
        .arg(&a)
        .arg(&b)
        .arg("-o")
        .arg(&merged)
        .output()
        .unwrap();
    assert!(
        ok.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let back: RuffleState =
        serde_json::from_str(&std::fs::read_to_string(&merged).unwrap()).unwrap();
    assert!((back.channels["x"].separation.count() - 4.0).abs() < 1e-9);

    // Incompatible: a tag swap under the kept key. Loud refusal, non-zero exit, no file.
    let c = dir.path().join("c.state");
    let nope = dir.path().join("nope.state");
    dump(&c, &state_with("model-2", &[9.0]));

    let refused = Command::new(bin)
        .args(["reconcile"])
        .arg(&a)
        .arg(&c)
        .arg("-o")
        .arg(&nope)
        .output()
        .unwrap();
    assert!(!refused.status.success(), "a tag swap must exit non-zero");
    assert!(!nope.exists(), "a refused merge must not write output");
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(stderr.contains("tag mismatch"), "stderr: {stderr}");
    assert!(stderr.contains('x'), "names the channel: {stderr}");
}
