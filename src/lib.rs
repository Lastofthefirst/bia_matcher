pub mod document_matcher;
pub mod error;
pub mod models;

// Re-export key types
pub use document_matcher::DocumentMatcher;
pub use models::{Config, DocumentMatch, PdfBlock, XmlElement, MatchResult};