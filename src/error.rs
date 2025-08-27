use thiserror::Error;

#[derive(Error, Debug)]
pub enum MatchingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Model loading error: {0}")]
    ModelLoading(String),
    
    #[error("Embedding generation error: {0}")]
    EmbeddingGeneration(String),
    
    #[error("FAISS error: {0}")]
    Faiss(String),
    
    #[error("No XML documents found")]
    NoXmlDocuments,
    
    #[error("No PDF documents found")]
    NoPdfDocuments,
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}