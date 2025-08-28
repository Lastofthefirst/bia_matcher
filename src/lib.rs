pub mod document_matcher;
pub mod error;
pub mod models;
pub mod embedding_service;
pub mod similarity_search;

// Re-export key types
pub use document_matcher::DocumentMatcher;
pub use models::{Config, DocumentMatch, PdfBlock, XmlElement, MatchResult};
pub use embedding_service::EmbeddingService;
pub use similarity_search::{DocumentSimilaritySearch, ParagraphSimilaritySearch};