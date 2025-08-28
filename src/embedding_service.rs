use anyhow::Result;
use reqwest;
use std::path::Path;
use std::fs;
use tokio::io::AsyncWriteExt;
use futures_util::stream::StreamExt;
use tracing::{info, warn};

const MODEL_URL: &str = "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-embed-text-v2-moe.Q2_K.gguf?download=true";
const MATRYOSHKA_DIMENSIONS: usize = 256;

pub struct EmbeddingService {
    _model_path: std::path::PathBuf,
}

impl EmbeddingService {
    pub async fn new(model_path: &Path) -> Result<Self> {
        info!("Initializing embedding service with model: {:?}", model_path);
        
        // Download model if it doesn't exist
        if !model_path.exists() {
            info!("Model not found locally, downloading from HuggingFace...");
            Self::download_model(model_path).await?;
        }
        
        // For now, we'll use a simplified approach
        // In production, you would load the actual model here
        info!("Embedding service initialized successfully (using placeholder implementation)");
        Ok(EmbeddingService { 
            _model_path: model_path.to_path_buf(),
        })
    }
    
    async fn download_model(model_path: &Path) -> Result<()> {
        info!("Downloading model from: {}", MODEL_URL);
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = model_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Download the model
        let response = reqwest::get(MODEL_URL).await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to download model: HTTP {}", response.status()));
        }
        
        let total_size = response.content_length().unwrap_or(0);
        info!("Model size: {} bytes", total_size);
        
        let mut file = tokio::fs::File::create(model_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            
            if total_size > 0 && downloaded % (total_size / 10).max(1024 * 1024) == 0 {
                info!("Downloaded {:.1}%", (downloaded as f64 / total_size as f64) * 100.0);
            }
        }
        
        file.sync_all().await?;
        info!("Model downloaded successfully to: {:?}", model_path);
        Ok(())
    }
    
    pub fn generate_embedding(&mut self, text: &str, prefix: &str) -> Result<Vec<f32>> {
        let _prefixed_text = format!("{}{}", prefix, text);
        
        // Generate a deterministic embedding based on text content
        // This is a placeholder - in production you'd use the actual model
        let mut embedding_vec = Vec::with_capacity(MATRYOSHKA_DIMENSIONS);
        
        // Simple hash-based embedding for consistent results
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let base_hash = hasher.finish();
        
        for i in 0..MATRYOSHKA_DIMENSIONS {
            let mut item_hasher = DefaultHasher::new();
            (base_hash, i).hash(&mut item_hasher);
            let val = (item_hasher.finish() as f32) / (u64::MAX as f32);
            embedding_vec.push(val * 2.0 - 1.0); // Normalize to [-1, 1]
        }
        
        // Normalize the vector
        let magnitude: f32 = embedding_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut embedding_vec {
                *val /= magnitude;
            }
        }
        
        Ok(embedding_vec)
    }
    
    pub fn generate_document_embedding(&mut self, text: &str) -> Result<Vec<f32>> {
        self.generate_embedding(text, "search_document: ")
    }
    
    pub fn generate_query_embedding(&mut self, text: &str) -> Result<Vec<f32>> {
        self.generate_embedding(text, "search_query: ")
    }
    
    pub fn embedding_dimension() -> usize {
        MATRYOSHKA_DIMENSIONS
    }
}