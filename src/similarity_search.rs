use crate::error::MatchingError;
use anyhow::Result;
use std::collections::HashMap;
use tracing::debug;

pub struct SimilaritySearch {
    embeddings: Vec<Vec<f32>>,
    id_mapping: Vec<String>, // Maps indices to document/element IDs
    embedding_dim: usize,
}

impl SimilaritySearch {
    pub fn new(embedding_dim: usize) -> Result<Self> {
        Ok(SimilaritySearch {
            embeddings: Vec::new(),
            id_mapping: Vec::new(),
            embedding_dim,
        })
    }
    
    pub fn add_embeddings(&mut self, embeddings: &[Vec<f32>], ids: &[String]) -> Result<()> {
        if embeddings.len() != ids.len() {
            return Err(MatchingError::Faiss("Mismatch between embeddings and IDs".to_string()).into());
        }
        
        if embeddings.is_empty() {
            return Ok(());
        }
        
        // Validate embedding dimensions
        for embedding in embeddings {
            if embedding.len() != self.embedding_dim {
                return Err(MatchingError::Faiss(format!(
                    "Embedding dimension mismatch: expected {}, got {}", 
                    self.embedding_dim, 
                    embedding.len()
                )).into());
            }
        }
        
        // Add embeddings and IDs
        for (embedding, id) in embeddings.iter().zip(ids.iter()) {
            self.embeddings.push(embedding.clone());
            self.id_mapping.push(id.clone());
        }
        
        debug!("Added {} embeddings to index", embeddings.len());
        Ok(())
    }
    
    pub fn search(&mut self, query_embedding: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if query_embedding.len() != self.embedding_dim {
            return Err(MatchingError::Faiss(format!(
                "Query embedding dimension mismatch: expected {}, got {}", 
                self.embedding_dim, 
                query_embedding.len()
            )).into());
        }
        
        if self.id_mapping.is_empty() {
            return Ok(Vec::new());
        }
        
        let k = k.min(self.id_mapping.len());
        
        // Calculate cosine similarity with all embeddings
        let mut similarities = Vec::new();
        for (i, embedding) in self.embeddings.iter().enumerate() {
            let similarity = cosine_similarity(query_embedding, embedding);
            similarities.push((i, similarity));
        }
        
        // Sort by similarity (highest first) and take top k
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        
        // Convert to (ID, similarity) pairs
        let results: Vec<(String, f32)> = similarities
            .into_iter()
            .map(|(idx, similarity)| (self.id_mapping[idx].clone(), similarity))
            .collect();
        
        debug!("Search returned {} results", results.len());
        Ok(results)
    }
    
    pub fn size(&self) -> usize {
        self.id_mapping.len()
    }
}

// Helper function to calculate cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

pub struct DocumentSimilaritySearch {
    search: SimilaritySearch,
    document_embeddings: HashMap<String, Vec<f32>>,
}

impl DocumentSimilaritySearch {
    pub fn new(embedding_dim: usize) -> Result<Self> {
        Ok(DocumentSimilaritySearch {
            search: SimilaritySearch::new(embedding_dim)?,
            document_embeddings: HashMap::new(),
        })
    }
    
    pub fn add_document(&mut self, document_id: String, embedding: Vec<f32>) -> Result<()> {
        self.document_embeddings.insert(document_id.clone(), embedding.clone());
        self.search.add_embeddings(&[embedding], &[document_id])?;
        Ok(())
    }
    
    pub fn find_best_document(&mut self, query_embedding: &[f32]) -> Result<Option<(String, f32)>> {
        let results = self.search.search(query_embedding, 1)?;
        Ok(results.into_iter().next())
    }
    
    pub fn get_all_document_similarities(&mut self, query_embedding: &[f32]) -> Result<Vec<(String, f32)>> {
        let all_documents = self.search.size();
        let mut results = self.search.search(query_embedding, all_documents)?;
        
        // Sort by similarity score (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(results)
    }
}

pub struct ParagraphSimilaritySearch {
    search: SimilaritySearch,
    element_embeddings: HashMap<String, Vec<f32>>,
}

impl ParagraphSimilaritySearch {
    pub fn new(embedding_dim: usize) -> Result<Self> {
        Ok(ParagraphSimilaritySearch {
            search: SimilaritySearch::new(embedding_dim)?,
            element_embeddings: HashMap::new(),
        })
    }
    
    pub fn add_elements(&mut self, elements: &[(String, Vec<f32>)]) -> Result<()> {
        if elements.is_empty() {
            return Ok(());
        }
        
        let ids: Vec<String> = elements.iter().map(|(id, _)| id.clone()).collect();
        let embeddings: Vec<Vec<f32>> = elements.iter().map(|(_, emb)| emb.clone()).collect();
        
        for (id, embedding) in elements {
            self.element_embeddings.insert(id.clone(), embedding.clone());
        }
        
        self.search.add_embeddings(&embeddings, &ids)?;
        Ok(())
    }
    
    pub fn find_matches(&mut self, query_embedding: &[f32], k: usize, threshold: f32) -> Result<Vec<(String, f32)>> {
        let results = self.search.search(query_embedding, k)?;
        
        // Filter by threshold
        let filtered_results: Vec<(String, f32)> = results
            .into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect();
        
        Ok(filtered_results)
    }
}