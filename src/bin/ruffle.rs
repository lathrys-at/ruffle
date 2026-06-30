//! The `ruffle` CLI: a thin wrapper over the state reconciliation API (§8, §11).
//!
//! Two subcommands, each a direct call into [`ruffle::RuffleState`]:
//!
//! - `reconcile` reads several state files, runs [`RuffleState::merge`] under
//!   [`MergePolicy::Strict`], and writes the merged state. It REFUSES (non-zero exit,
//!   no output written) on any format / fingerprint / direction / tag mismatch, because
//!   a model swap under a kept tag must fail loudly rather than silently blend two
//!   distributions (§8). It WARNS, non-fatally, on a channel present in some inputs but
//!   not all: those channels still carry through the union, and the warning only lets
//!   the operator notice an asymmetric input set.
//! - `rekey` rewrites one channel's key, the safe rename path (§8), and writes the
//!   result.
//!
//! All algorithmic decisions live in `RuffleState`. This binary parses arguments,
//! moves bytes to and from disk, and reports. The divergence it prints under `--report`
//! is advisory: it never overrides the tag gate (§8).

use clap::{Args, Parser, Subcommand};
use ruffle::{Divergence, MergePolicy, Mismatch, RuffleState};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "ruffle",
    about = "Reconcile and rename ruffle persistent state (§8).",
    version,
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Merge several state files into one, refusing on any incompatibility.
    Reconcile(ReconcileArgs),
    /// Rename a channel's key within a state file (the safe rename path).
    Rekey(RekeyArgs),
}

#[derive(Args)]
struct ReconcileArgs {
    /// Input state files (one or more). A single input is canonicalised in place.
    #[arg(required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,
    /// Where to write the merged state (canonical JSON).
    #[arg(short = 'o', long = "output")]
    output: PathBuf,
    /// Also print the advisory divergence and a per-channel merged/carried summary.
    #[arg(long)]
    report: bool,
}

#[derive(Args)]
struct RekeyArgs {
    /// The state file to rewrite.
    state: PathBuf,
    /// The channel key to rename from.
    #[arg(long)]
    from: String,
    /// The channel key to rename to.
    #[arg(long)]
    to: String,
    /// Where to write the rekeyed state (canonical JSON).
    #[arg(short = 'o', long = "output")]
    output: PathBuf,
    /// Run divergence of the result against the input as a sanity check and print it.
    #[arg(long)]
    report: bool,
}

/// An operator-facing failure. Every variant carries enough to say which file or which
/// channel is at fault, so the message is actionable on its own.
#[derive(Debug)]
enum CliError {
    /// A state file could not be read.
    Read(PathBuf, io::Error),
    /// A state file could not be parsed as a `RuffleState`.
    Parse(PathBuf, serde_json::Error),
    /// The output file could not be written.
    Write(PathBuf, io::Error),
    /// The merged state could not be serialized.
    Encode(serde_json::Error),
    /// The merge was refused: incompatible inputs (§8). No output is written.
    Refused(Mismatch),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Read(path, e) => {
                write!(f, "cannot read state file {}: {e}", path.display())
            }
            CliError::Parse(path, e) => {
                write!(f, "cannot parse {} as a ruffle state: {e}", path.display())
            }
            CliError::Write(path, e) => {
                write!(f, "cannot write {}: {e}", path.display())
            }
            CliError::Encode(e) => write!(f, "cannot serialize state: {e}"),
            // The Mismatch Display already names the channel / versions; prefix it so the
            // refusal reads as a deliberate stop, not an internal failure.
            CliError::Refused(m) => write!(f, "refusing to merge incompatible states: {m}"),
        }
    }
}

impl std::error::Error for CliError {}

/// Read and parse one state file.
fn load_state(path: &Path) -> Result<RuffleState, CliError> {
    let text = fs::read_to_string(path).map_err(|e| CliError::Read(path.to_path_buf(), e))?;
    serde_json::from_str(&text).map_err(|e| CliError::Parse(path.to_path_buf(), e))
}

/// Serialize a state to canonical JSON and write it. The `BTreeMap` ordering in
/// `RuffleState` makes this output content-addressable: identical states write
/// byte-identical files (§8).
fn write_state(state: &RuffleState, output: &Path) -> Result<(), CliError> {
    let json = serde_json::to_string(state).map_err(CliError::Encode)?;
    fs::write(output, json).map_err(|e| CliError::Write(output.to_path_buf(), e))
}

/// How many of the inputs carry each channel key.
fn channel_input_counts(parts: &[RuffleState]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for part in parts {
        for key in part.channels.keys() {
            *counts.entry(key.clone()).or_insert(0) += 1;
        }
    }
    counts
}

/// Reconcile several state files into one.
///
/// Refuses (returns `Err`, writes nothing) on any incompatibility. On success the merged
/// state is written to `output`; only then are warnings and the optional report emitted,
/// so a refused or failed run produces no advisory noise. Warnings go to `err`, the
/// report to `out`.
fn reconcile(
    inputs: &[PathBuf],
    output: &Path,
    report: bool,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<(), CliError> {
    let parts = inputs
        .iter()
        .map(|p| load_state(p))
        .collect::<Result<Vec<_>, _>>()?;
    let refs: Vec<&RuffleState> = parts.iter().collect();

    // The single merge of §8. Divergence comes back alongside the merged state; the gate
    // is the tag/fingerprint/format check inside merge, never the divergence.
    let (merged, divergence) =
        RuffleState::merge(&refs, MergePolicy::Strict).map_err(CliError::Refused)?;

    write_state(&merged, output)?;

    // Warn on asymmetric inputs: a channel in some inputs but not all still carries
    // through the union, but the operator should see that the input set was uneven.
    let counts = channel_input_counts(&parts);
    let n = parts.len();
    for (key, &count) in &counts {
        if count < n {
            let _ = writeln!(
                err,
                "ruffle: warning: channel \"{key}\" present in {count} of {n} inputs; \
                 carried through the union, not dropped"
            );
        }
    }

    if report {
        let _ = write_reconcile_report(out, &divergence, &counts);
    }

    Ok(())
}

/// Print the advisory divergence plus which channels merged versus carried.
fn write_reconcile_report(
    out: &mut dyn Write,
    divergence: &Divergence,
    counts: &BTreeMap<String, usize>,
) -> io::Result<()> {
    writeln!(
        out,
        "divergence (advisory; never overrides the tag gate). High divergence under \
         matching tags is the silent-model-swap signature; low divergence under \
         differing tags is benign."
    )?;
    writeln!(out, "  max: {:.6}", divergence.max)?;
    for (key, d) in &divergence.per_channel {
        writeln!(out, "  {key}: {d:.6}")?;
    }

    let merged: Vec<String> = counts
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(k, _)| k.to_string())
        .collect();
    let carried: Vec<String> = counts
        .iter()
        .filter(|(_, count)| **count == 1)
        .map(|(k, _)| k.to_string())
        .collect();
    writeln!(
        out,
        "channels merged (present in more than one input): {}",
        if merged.is_empty() {
            "(none)".to_string()
        } else {
            merged.join(", ")
        }
    )?;
    writeln!(
        out,
        "channels carried from a single input: {}",
        if carried.is_empty() {
            "(none)".to_string()
        } else {
            carried.join(", ")
        }
    )?;
    Ok(())
}

/// Rename a channel key in a state file, the safe rename path (§8).
///
/// With `report`, divergence of the rewritten state against the input is printed as a
/// sanity check: the renamed channel drops out of the comparison (its key no longer
/// matches), and every untouched channel should read zero.
fn rekey_cmd(
    input: &Path,
    from: &str,
    to: &str,
    output: &Path,
    report: bool,
    out: &mut dyn Write,
) -> Result<(), CliError> {
    let original = load_state(input)?;
    let mut state = original.clone();
    state.rekey(from, to.to_string());
    write_state(&state, output)?;

    if report {
        let divergence = state.divergence(&original);
        let _ = writeln!(
            out,
            "rekey sanity check: divergence of the rekeyed state against the input \
             (the renamed channel drops out; untouched channels should read 0)."
        );
        let _ = writeln!(out, "  max: {:.6}", divergence.max);
        for (key, d) in &divergence.per_channel {
            let _ = writeln!(out, "  {key}: {d:.6}");
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut out = io::stdout();
    let mut err = io::stderr();

    let result = match cli.command {
        Command::Reconcile(args) => {
            reconcile(&args.inputs, &args.output, args.report, &mut out, &mut err)
        }
        Command::Rekey(args) => rekey_cmd(
            &args.state,
            &args.from,
            &args.to,
            &args.output,
            args.report,
            &mut out,
        ),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(err, "ruffle: error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruffle::{
        BaselineMode, ChannelSummary, Direction, PairSummary, StatFingerprint, UnorderedPair,
    };
    use tempfile::tempdir;

    // --- builders ---

    fn fingerprint(dirs: &[(&str, Direction)]) -> StatFingerprint {
        let mut m = BTreeMap::new();
        for (k, d) in dirs {
            m.insert((*k).to_string(), *d);
        }
        StatFingerprint::new(BaselineMode::ZScore, m)
    }

    /// A channel summary tagged `tag`, its separation baseline built from `sep`.
    fn chan(tag: &str, sep: &[f64]) -> ChannelSummary {
        let mut c = ChannelSummary::new(tag.to_string());
        for &x in sep {
            c.separation.push(x);
        }
        c
    }

    fn state(fp: StatFingerprint, chans: &[(&str, ChannelSummary)]) -> RuffleState {
        let mut s = RuffleState::new(fp);
        for (k, c) in chans {
            s.channels.insert((*k).to_string(), c.clone());
        }
        s
    }

    fn dump(dir: &Path, name: &str, s: &RuffleState) -> PathBuf {
        let p = dir.join(name);
        write_state(s, &p).unwrap();
        p
    }

    fn slurp(p: &Path) -> RuffleState {
        load_state(p).unwrap()
    }

    use Direction::{HigherIsBetter, LowerIsBetter};

    // --- reconcile: the happy path round-trips and matches the library merge ---

    #[test]
    fn reconcile_merges_and_matches_library_merge() {
        let dir = tempdir().unwrap();
        let a = state(
            fingerprint(&[("x", HigherIsBetter), ("y", HigherIsBetter)]),
            &[
                ("x", chan("m", &[1.0, 2.0, 3.0])),
                ("y", chan("m", &[4.0, 5.0])),
            ],
        );
        let b = state(
            fingerprint(&[("x", HigherIsBetter), ("z", LowerIsBetter)]),
            &[("x", chan("m", &[10.0, 11.0])), ("z", chan("m", &[6.0]))],
        );
        let pa = dump(dir.path(), "a.state", &a);
        let pb = dump(dir.path(), "b.state", &b);
        let outp = dir.path().join("merged.state");

        let (mut out, mut err) = (Vec::new(), Vec::new());
        reconcile(&[pa, pb], &outp, false, &mut out, &mut err).unwrap();

        let got = slurp(&outp);
        let (expected, _) = RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap();
        assert_eq!(got, expected, "written state equals the library merge");
        // x was in both inputs, so its baseline pooled all five observations.
        assert!((got.channels["x"].separation.count() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn reconcile_output_round_trips_byte_identically() {
        let dir = tempdir().unwrap();
        let a = state(
            fingerprint(&[("x", HigherIsBetter)]),
            &[("x", chan("m", &[1.0, 2.0]))],
        );
        let b = state(
            fingerprint(&[("x", HigherIsBetter)]),
            &[("x", chan("m", &[3.0, 4.0]))],
        );
        let pa = dump(dir.path(), "a.state", &a);
        let pb = dump(dir.path(), "b.state", &b);
        let outp = dir.path().join("merged.state");

        let (mut out, mut err) = (Vec::new(), Vec::new());
        reconcile(&[pa, pb], &outp, false, &mut out, &mut err).unwrap();

        // Re-serializing the parsed-back state reproduces the file: canonical output.
        let on_disk = fs::read_to_string(&outp).unwrap();
        let reparsed = serde_json::to_string(&slurp(&outp)).unwrap();
        assert_eq!(on_disk, reparsed);
    }

    // --- reconcile: every incompatibility is a loud refusal that writes nothing ---

    fn assert_refuses(parts: &[(&str, RuffleState)], expect: &Mismatch, needles: &[&str]) {
        let dir = tempdir().unwrap();
        let paths: Vec<PathBuf> = parts
            .iter()
            .map(|(name, s)| dump(dir.path(), name, s))
            .collect();
        let outp = dir.path().join("must-not-exist.state");

        let (mut out, mut err) = (Vec::new(), Vec::new());
        let e = reconcile(&paths, &outp, false, &mut out, &mut err).unwrap_err();

        match &e {
            CliError::Refused(m) => assert_eq!(m, expect, "refusal kind"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        let msg = e.to_string();
        for needle in needles {
            assert!(
                msg.contains(needle),
                "message {msg:?} should mention {needle:?}"
            );
        }
        assert!(!outp.exists(), "a refused merge must not write output");
    }

    #[test]
    fn reconcile_refuses_on_tag_mismatch() {
        let a = state(
            fingerprint(&[("x", HigherIsBetter)]),
            &[("x", chan("model-1", &[1.0]))],
        );
        let b = state(
            fingerprint(&[("x", HigherIsBetter)]),
            &[("x", chan("model-2", &[2.0]))],
        );
        assert_refuses(
            &[("a.state", a), ("b.state", b)],
            &Mismatch::Tag {
                channel: "x".into(),
                left: "model-1".into(),
                right: "model-2".into(),
            },
            &["tag mismatch", "x", "model-1", "model-2"],
        );
    }

    #[test]
    fn reconcile_refuses_on_format_version_mismatch() {
        let a = state(fingerprint(&[]), &[]);
        // `format_version` is library-managed with no setter; produce the mismatched
        // state the way a loaded file would carry it — edit the serialized value.
        let mut b_value = serde_json::to_value(state(fingerprint(&[]), &[])).unwrap();
        b_value["format_version"] = serde_json::Value::from(99u32);
        let b: RuffleState = serde_json::from_value(b_value).unwrap();
        assert_refuses(
            &[("a.state", a), ("b.state", b)],
            &Mismatch::FormatVersion {
                left: RuffleState::FORMAT_VERSION,
                right: 99,
            },
            &["format version mismatch", "99"],
        );
    }

    #[test]
    fn reconcile_refuses_on_fingerprint_mismatch() {
        let a = state(fingerprint(&[]), &[]);
        // Set the mismatching `stat_version` on the `StatFingerprint` (its own fields stay
        // public) before building the state.
        let mut fp = fingerprint(&[]);
        fp.stat_version = 99;
        let b = state(fp, &[]);
        assert_refuses(
            &[("a.state", a), ("b.state", b)],
            &Mismatch::Fingerprint,
            &["fingerprint mismatch"],
        );
    }

    #[test]
    fn reconcile_refuses_on_direction_conflict() {
        let a = state(fingerprint(&[("x", HigherIsBetter)]), &[]);
        let b = state(fingerprint(&[("x", LowerIsBetter)]), &[]);
        assert_refuses(
            &[("a.state", a), ("b.state", b)],
            &Mismatch::DirectionConflict {
                channel: "x".into(),
            },
            &["direction conflict", "x"],
        );
    }

    // --- reconcile: asymmetric inputs warn but still carry the channel through ---

    #[test]
    fn reconcile_warns_on_asymmetric_key_but_merges_it() {
        let dir = tempdir().unwrap();
        let a = state(
            fingerprint(&[("x", HigherIsBetter), ("y", HigherIsBetter)]),
            &[("x", chan("m", &[1.0, 2.0])), ("y", chan("m", &[7.0]))],
        );
        let b = state(
            fingerprint(&[("x", HigherIsBetter)]),
            &[("x", chan("m", &[3.0]))],
        );
        let pa = dump(dir.path(), "a.state", &a);
        let pb = dump(dir.path(), "b.state", &b);
        let outp = dir.path().join("merged.state");

        let (mut out, mut err) = (Vec::new(), Vec::new());
        reconcile(&[pa, pb], &outp, false, &mut out, &mut err).unwrap();

        let warn = String::from_utf8(err).unwrap();
        assert!(warn.contains("warning"), "warning text: {warn:?}");
        assert!(
            warn.contains("\"y\""),
            "names the asymmetric channel: {warn:?}"
        );
        assert!(warn.contains("1 of 2"), "states the input count: {warn:?}");
        // y is present in only one input, but it is NOT dropped.
        let got = slurp(&outp);
        assert!(got.channels.contains_key("y"));
        assert!(got.channels.contains_key("x"));
    }

    #[test]
    fn reconcile_report_prints_divergence_and_partition() {
        let dir = tempdir().unwrap();
        // x in both (and shifted, so divergence is large); y only in a.
        let a = state(
            fingerprint(&[("x", HigherIsBetter), ("y", HigherIsBetter)]),
            &[
                ("x", chan("m", &[-1.0, 0.0, 1.0])),
                ("y", chan("m", &[1.0, 2.0])),
            ],
        );
        let b = state(
            fingerprint(&[("x", HigherIsBetter)]),
            &[("x", chan("m", &[9.0, 10.0, 11.0]))],
        );
        let pa = dump(dir.path(), "a.state", &a);
        let pb = dump(dir.path(), "b.state", &b);
        let outp = dir.path().join("merged.state");

        let (mut out, mut err) = (Vec::new(), Vec::new());
        reconcile(&[pa, pb], &outp, true, &mut out, &mut err).unwrap();

        let report = String::from_utf8(out).unwrap();
        assert!(report.contains("divergence"), "report: {report:?}");
        assert!(report.contains("max:"), "report: {report:?}");
        // x merged (in both), y carried (single input) -- each named on the right
        // line and only there, so the partition itself is pinned, not just the labels.
        let merged_line = report
            .lines()
            .find(|l| l.starts_with("channels merged"))
            .expect("merged line present");
        assert!(
            merged_line.ends_with(": x"),
            "exactly x merged: {merged_line:?}"
        );
        let carried_line = report
            .lines()
            .find(|l| l.starts_with("channels carried"))
            .expect("carried line present");
        assert!(
            carried_line.ends_with(": y"),
            "exactly y carried: {carried_line:?}"
        );
    }

    #[test]
    fn reconcile_single_input_canonicalises() {
        let dir = tempdir().unwrap();
        let a = state(
            fingerprint(&[("x", HigherIsBetter)]),
            &[("x", chan("m", &[1.0, 2.0, 3.0]))],
        );
        let pa = dump(dir.path(), "a.state", &a);
        let outp = dir.path().join("out.state");

        let (mut out, mut err) = (Vec::new(), Vec::new());
        reconcile(&[pa], &outp, false, &mut out, &mut err).unwrap();
        assert_eq!(slurp(&outp), a);
        // No asymmetry warning for a single input.
        assert!(String::from_utf8(err).unwrap().is_empty());
    }

    #[test]
    fn reconcile_reports_unreadable_input() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope.state");
        let outp = dir.path().join("out.state");
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let e = reconcile(&[missing], &outp, false, &mut out, &mut err).unwrap_err();
        assert!(matches!(e, CliError::Read(..)));
        assert!(!outp.exists());
    }

    #[test]
    fn reconcile_reports_unparseable_input() {
        let dir = tempdir().unwrap();
        let bad = dir.path().join("bad.state");
        fs::write(&bad, "{ not valid ruffle state ]").unwrap();
        let outp = dir.path().join("out.state");
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let e = reconcile(&[bad], &outp, false, &mut out, &mut err).unwrap_err();
        assert!(matches!(e, CliError::Parse(..)));
        assert!(!outp.exists());
    }

    // --- rekey: the safe rename round-trips ---

    #[test]
    fn rekey_renames_and_round_trips() {
        let dir = tempdir().unwrap();
        let mut s = state(
            fingerprint(&[("old", LowerIsBetter), ("keep", HigherIsBetter)]),
            &[
                ("old", chan("m", &[1.0, 2.0, 3.0])),
                ("keep", chan("m", &[4.0])),
            ],
        );
        let mut pair = PairSummary::new();
        pair.redundancy.push(0.5);
        s.pairs.insert(
            UnorderedPair::new("old".to_string(), "keep".to_string()),
            pair,
        );
        let old_mean = s.channels["old"].separation.mean();

        let p = dump(dir.path(), "in.state", &s);
        let outp = dir.path().join("renamed.state");
        let mut out = Vec::new();
        rekey_cmd(&p, "old", "new", &outp, false, &mut out).unwrap();

        let got = slurp(&outp);
        assert!(!got.channels.contains_key("old"));
        assert!(got.channels.contains_key("new"));
        assert!((got.channels["new"].separation.mean() - old_mean).abs() < 1e-12);
        // The pair and the orientation followed the rename.
        assert!(
            got.pairs
                .contains_key(&UnorderedPair::new("new".to_string(), "keep".to_string()))
        );
        assert_eq!(
            got.fingerprint().directions.get("new"),
            Some(&LowerIsBetter)
        );
    }

    #[test]
    fn rekey_report_is_zero_for_untouched_channels() {
        let dir = tempdir().unwrap();
        let s = state(
            fingerprint(&[("old", HigherIsBetter), ("keep", HigherIsBetter)]),
            &[
                ("old", chan("m", &[1.0, 2.0])),
                ("keep", chan("m", &[5.0, 6.0, 7.0])),
            ],
        );
        let p = dump(dir.path(), "in.state", &s);
        let outp = dir.path().join("renamed.state");
        let mut out = Vec::new();
        rekey_cmd(&p, "old", "new", &outp, true, &mut out).unwrap();

        let report = String::from_utf8(out).unwrap();
        assert!(report.contains("sanity check"), "report: {report:?}");
        // `keep` is unchanged, so its divergence reads zero.
        assert!(report.contains("keep: 0.000000"), "report: {report:?}");
    }
}
