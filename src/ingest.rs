//! Ingest: one query's inputs and the coupling anchor, in canonical units (§5.3, §7).
//!
//! Everything a caller feeds in enters through here, oriented to higher-is-better and
//! sanitized:
//!
//! - [`input`]: one query's input for one channel (§7).
//! - [`anchor`]: the full-scored anchor coupling reads its redundancy from (§5.3).

pub mod anchor;
pub mod input;
