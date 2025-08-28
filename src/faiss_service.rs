use anyhow::Result;
use tracing::{info, warn};

pub struct FaissService {
    document_embeddings: Vec<Vec<f32>>,
    paragraph_embeddings: Vec<Vec<f32>>,
    document_ids: Vec<String>,
    paragraph_ids: Vec<String>,
    dimension: usize,
}

impl FaissService {
    pub fn new(dimension: usize) -> Result<Self> {
        info!("Initializing FAISS service with dimension: {}", dimension);
        
        Ok(FaissService {
            document_embeddings: Vec::new(),
            paragraph_embeddings: Vec::new(),
            document_ids: Vec::new(),
            paragraph_ids: Vec::new(),
            dimension,
        })
    }
    
    pub fn add_document_embeddings(&mut self, embeddings: &[Vec<f32>], ids: &[String]) -> Result<()> {
        if embeddings.len() != ids.len() {
            return Err(anyhow::anyhow!("Embeddings and IDs length mismatch"));
        }
        
        if embeddings.is_empty() {
            return Ok(());
        }
        
        // Store embeddings and IDs
        self.document_embeddings.extend(embeddings.iter().cloned());
        self.document_ids.extend(ids.iter().cloned());
        
        info!("Added {} document embeddings to index", embeddings.len());
        Ok(())
    }
    
    pub fn add_paragraph_embeddings(&mut self, embeddings: &[Vec<f32>], ids: &[String]) -> Result<()> {
        if embeddings.len() != ids.len() {
            return Err(anyhow::anyhow!("Embeddings and IDs length mismatch"));
        }
        
        if embeddings.is_empty() {
            return Ok(());
        }
        
        // Store embeddings and IDs
        self.paragraph_embeddings.extend(embeddings.iter().cloned());
        self.paragraph_ids.extend(ids.iter().cloned());
        
        info!("Added {} paragraph embeddings to index", embeddings.len());
        Ok(())
    }
    
    pub fn search_documents(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if self.document_embeddings.is_empty() {
            warn!("Document index is empty");
            return Ok(Vec::new());
        }
        
        let mut similarities = Vec::new();
        
        // Calculate cosine similarity with all document embeddings
        for (i, doc_embedding) in self.document_embeddings.iter().enumerate() {
            let similarity = cosine_similarity(query_embedding, doc_embedding);
            similarities.push((i, similarity));
        }
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top k results
        let results: Vec<(String, f32)> = similarities
            .into_iter()
            .take(k)
            .map(|(i, similarity)| (self.document_ids[i].clone(), similarity))
            .collect();
        
        Ok(results)
    }
    
    pub fn search_paragraphs(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if self.paragraph_embeddings.is_empty() {
            warn!("Paragraph index is empty");
            return Ok(Vec::new());
        }
        
        let mut similarities = Vec::new();
        
        // Calculate cosine similarity with all paragraph embeddings
        for (i, para_embedding) in self.paragraph_embeddings.iter().enumerate() {
            let similarity = cosine_similarity(query_embedding, para_embedding);
            similarities.push((i, similarity));
        }
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top k results
        let results: Vec<(String, f32)> = similarities
            .into_iter()
            .take(k)
            .map(|(i, similarity)| (self.paragraph_ids[i].clone(), similarity))
            .collect();
        
        Ok(results)
    }
    
    pub fn clear_document_index(&mut self) {
        self.document_embeddings.clear();
        self.document_ids.clear();
        info!("Cleared document index");
    }
    
    pub fn clear_paragraph_index(&mut self) {
        self.paragraph_embeddings.clear();
        self.paragraph_ids.clear();
        info!("Cleared paragraph index");
    }
    
    pub fn document_count(&self) -> usize {
        self.document_ids.len()
    }
    
    pub fn paragraph_count(&self) -> usize {
        self.paragraph_ids.len()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm_a * norm_b)
}