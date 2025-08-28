pub mod document_matcher;
pub mod embedding_service;
pub mod faiss_service;
pub mod error;
pub mod models;

// Re-export key types
pub use document_matcher::DocumentMatcher;
pub use embedding_service::EmbeddingService;
pub use faiss_service::FaissService;
pub use models::{Config, DocumentMatch, PdfBlock, XmlElement, MatchResult};