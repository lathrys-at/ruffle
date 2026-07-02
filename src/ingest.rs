//! Ingest: one query's inputs and the coupling anchor, in canonical units.
//!
//! Everything a caller feeds in enters through here, oriented to higher-is-better and
//! sanitized:
//!
//! - [`input`]: one query's input for one channel.
//! - [`anchor`]: the full-scored anchor coupling reads its redundancy from.

pub mod anchor;
pub mod input;
