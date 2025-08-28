use crate::error::MatchingError;
use crate::models::{Config, DocumentMatch, PdfBlock, XmlElement, ElementMatchResult, MatchResult, DocumentMatchInfo, MatchStatistics};
use anyhow::Result;
use serde_json;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::fs;
use std::path::Path;
use tracing::info;

pub struct DocumentMatcher {
    config: Config,
    xml_documents: HashMap<String, Vec<XmlElement>>,
    pdf_documents: HashMap<String, Vec<PdfBlock>>,
}

impl DocumentMatcher {
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing DocumentMatcher with embedding-based approach");
        
        Ok(DocumentMatcher {
            config,
            xml_documents: HashMap::new(),
            pdf_documents: HashMap::new(),
        })
    }
    
    pub async fn process_all_pdfs(&mut self) -> Result<()> {
        info!("Loading XML documents");
        self.load_xml_documents().await?;
        
        info!("Loading PDF documents");
        self.load_pdf_documents().await?;
        
        info!("Processing PDF documents for matching with embedding-based approach");
        self.match_documents().await?;
        
        Ok(())
    }
    
    async fn load_xml_documents(&mut self) -> Result<()> {
        let xml_dir = self.config.xml_documents_dir.clone();
        if !xml_dir.exists() {
            return Err(MatchingError::InvalidConfig(format!("XML documents directory does not exist: {:?}", xml_dir)).into());
        }
        
        self.load_xml_documents_recursive(&xml_dir)?;
        
        info!("Loaded {} XML documents", self.xml_documents.len());
        Ok(())
    }
    
    fn load_xml_documents_recursive(&mut self, dir: &Path) -> Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    // Recursively process subdirectories
                    self.load_xml_documents_recursive(&path)?;
                } else if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    // Process JSON files
                    info!("Loading XML document: {:?}", path);
                    let file_content = fs::read_to_string(&path)?;
                    let elements: Vec<XmlElement> = serde_json::from_str(&file_content)?;
                    
                    let filename = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    
                    self.xml_documents.insert(filename, elements);
                }
            }
        }
        Ok(())
    }
    
    async fn load_pdf_documents(&mut self) -> Result<()> {
        let pdf_dir = self.config.pdf_input_dir.clone();
        if !pdf_dir.exists() {
            return Err(MatchingError::InvalidConfig(format!("PDF input directory does not exist: {:?}", pdf_dir)).into());
        }
        
        self.load_pdf_documents_recursive(&pdf_dir)?;
        
        info!("Loaded {} PDF documents", self.pdf_documents.len());
        Ok(())
    }
    
    fn load_pdf_documents_recursive(&mut self, dir: &Path) -> Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    // Recursively process subdirectories
                    self.load_pdf_documents_recursive(&path)?;
                } else if path.extension().and_then(|s| s.to_str()) == Some("json") 
                    && path.to_string_lossy().ends_with("_processed.json") {
                    // Process PDF JSON files that end with "_processed.json"
                    info!("Loading PDF document: {:?}", path);
                    let file_content = fs::read_to_string(&path)?;
                    let blocks: Vec<PdfBlock> = serde_json::from_str(&file_content)?;
                    
                    let filename = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    
                    self.pdf_documents.insert(filename, blocks);
                }
            }
        }
        Ok(())
    }
    
    async fn match_documents(&mut self) -> Result<()> {
        if self.xml_documents.is_empty() {
            return Err(MatchingError::NoXmlDocuments.into());
        }
        
        if self.pdf_documents.is_empty() {
            return Err(MatchingError::NoPdfDocuments.into());
        }
        
        // Clone the PDF documents to avoid borrowing issues
        let pdf_documents = self.pdf_documents.clone();
        
        for (pdf_name, pdf_blocks) in &pdf_documents {
            info!("Matching PDF document: {}", pdf_name);
            
            // Calculate document-level similarities using embeddings
            let document_similarities = self.calculate_document_similarities_embedding(pdf_blocks);
            
            // Find the best matching XML document
            let best_match = document_similarities
                .iter()
                .max_by(|a, b| a.similarity.partial_cmp(&b.similarity).unwrap())
                .unwrap(); // Safe because we checked for empty xml_documents earlier
            
            info!("Best matching XML document for {}: {} (score: {})", 
                  pdf_name, best_match.xml_document_id, best_match.similarity);
            
            // Perform paragraph-level matching with the best XML document
            let best_xml_elements = self.xml_documents.get(&best_match.xml_document_id).unwrap().clone();
            let (matches, statistics) = self.embedding_match_paragraphs(&best_xml_elements, pdf_blocks);
            
            let document_match = DocumentMatch {
                pdf_document_id: pdf_name.clone(),
                best_match: (*best_match).clone(),
                all_document_scores: document_similarities,
                paragraph_matches: matches,
                statistics,
            };
            
            // Save results
            self.save_results(pdf_name, &document_match).await?;
        }
        
        Ok(())
    }
    
    fn calculate_document_similarities_embedding(&mut self, pdf_blocks: &[PdfBlock]) -> Vec<DocumentMatchInfo> {
        info!("Using embedding-based document similarity calculation");
        
        // Concatenate first 30 blocks for document-level comparison
        let pdf_doc_text: String = pdf_blocks.iter()
            .take(30)
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Generate hash-based embedding for PDF document (with "search_query:" prefix)
        let prefixed_pdf_text = format!("search_query: {}", pdf_doc_text);
        let pdf_embedding = self.generate_hash_embedding(&prefixed_pdf_text);
        
        let mut similarities = Vec::new();
        
        // Compare against all XML documents
        for (doc_id, xml_elements) in &self.xml_documents {
            // Concatenate first 30 elements for document-level representation
            let xml_doc_text: String = xml_elements.iter()
                .take(30)
                .map(|elem| elem.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            
            // Generate hash-based embedding for XML document (with "search_document:" prefix)
            let prefixed_xml_text = format!("search_document: {}", xml_doc_text);
            let xml_embedding = self.generate_hash_embedding(&prefixed_xml_text);
            
            // Calculate cosine similarity
            let similarity = cosine_similarity(&pdf_embedding, &xml_embedding);
            
            similarities.push(DocumentMatchInfo {
                xml_document_id: doc_id.clone(),
                similarity,
            });
        }
        
        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        similarities
    }
        let mut similarities = Vec::new();
        
        // Get first 30 blocks for document-level matching (as per specification)
        let first_blocks: Vec<&PdfBlock> = pdf_blocks.iter().take(30).collect();
        
        // Concatenate text from first blocks
        let concatenated_text: String = first_blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Calculate similarity with each XML document
        for (xml_doc_name, xml_elements) in &self.xml_documents {
            // For document-level matching, we'll use the first few XML elements as representatives
            let xml_texts: Vec<&str> = xml_elements.iter().take(10).map(|e| e.text.as_str()).collect();
            let concatenated_xml_text = xml_texts.join(" ");
            
            let similarity = self.calculate_similarity(&concatenated_text, &concatenated_xml_text);
            
            similarities.push(DocumentMatchInfo {
                xml_document_id: xml_doc_name.clone(),
                similarity,
            });
        }
        
        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        
        similarities
    }
    
    fn embedding_match_paragraphs(&mut self, xml_elements: &[XmlElement], pdf_blocks: &[PdfBlock]) -> (Vec<ElementMatchResult>, MatchStatistics) {
        info!("Using embedding-based paragraph matching");
        
        let mut results = Vec::new();
        let mut matched_elements_count = 0;
        
        // For each XML element, find the top K matching PDF blocks using embeddings
        for xml_element in xml_elements.iter().take(50) {  // Process up to 50 XML elements
            // Generate hash-based embedding for XML element (with "search_document:" prefix)
            let prefixed_xml_text = format!("search_document: {}", xml_element.text);
            let xml_embedding = self.generate_hash_embedding(&prefixed_xml_text);
            
            let mut element_matches = Vec::new();
            
            // Match against all PDF blocks using cosine similarity
            for pdf_block in pdf_blocks.iter() {
                // Generate hash-based embedding for PDF block (with "search_query:" prefix)
                let prefixed_pdf_text = format!("search_query: {}", pdf_block.text);
                let pdf_embedding = self.generate_hash_embedding(&prefixed_pdf_text);
                
                // Calculate cosine similarity
                let similarity = cosine_similarity(&xml_embedding, &pdf_embedding);
                
                // Store matches above threshold
                if similarity > self.config.similarity_threshold {
                    element_matches.push(MatchResult {
                        pdf_block_id: pdf_block.id.clone(),
                        similarity,
                        pdf_text: pdf_block.text.clone(),
                    });
                }
            }
            
            // Sort by similarity (highest first) and take top K
            element_matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
            element_matches.truncate(self.config.top_k_matches);
            
            // Count matched elements
            if !element_matches.is_empty() {
                matched_elements_count += 1;
            }
            
            // Create element match result
            let element_result = ElementMatchResult {
                xml_element_id: xml_element.id.clone(),
                xml_text: xml_element.text.clone(),
                top_matches: element_matches,
            };
            
            results.push(element_result);
        }
        
        // Create statistics
        let processed_xml_elements = xml_elements.len().min(50);
        let unmatched_elements = processed_xml_elements.saturating_sub(matched_elements_count);
        
        let statistics = MatchStatistics {
            total_xml_documents: self.xml_documents.len(),
            total_xml_elements: xml_elements.len(),
            total_pdf_blocks: pdf_blocks.len(),
            matched_elements: matched_elements_count,
            unmatched_elements,
            match_threshold: self.config.similarity_threshold,
            top_k_matches: self.config.top_k_matches,
        };
        
        (results, statistics)
    }
    
    
    fn generate_hash_embedding(&self, text: &str) -> Vec<f32> {
        const EMBEDDING_DIMENSION: usize = 256;
        
        // Generate a deterministic embedding based on text content
        let mut embedding_vec = Vec::with_capacity(EMBEDDING_DIMENSION);
        
        // Simple hash-based embedding for consistent results
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let base_hash = hasher.finish();
        
        for i in 0..EMBEDDING_DIMENSION {
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
        
        embedding_vec
    }
    
    fn simple_match_paragraphs(&self, xml_elements: &[XmlElement], pdf_blocks: &[PdfBlock]) -> (Vec<ElementMatchResult>, MatchStatistics) {
        let mut results = Vec::new();
        let mut matched_elements_count = 0;
        
        // For each XML element, find the top 3 matching PDF blocks
        for xml_element in xml_elements.iter().take(50) {  // Process up to 50 XML elements
            let mut element_matches = Vec::new();
            
            // Match this XML element against all PDF blocks
            for pdf_block in pdf_blocks.iter() {  // Process all PDF blocks
                let similarity = self.calculate_similarity(&xml_element.text, &pdf_block.text);
                
                // Store all matches above threshold
                if similarity > 0.01 {
                    element_matches.push(MatchResult {
                        pdf_block_id: pdf_block.id.clone(),
                        similarity,
                        pdf_text: pdf_block.text.clone(),
                    });
                }
            }
            
            // Sort by similarity (highest first) and take top 3
            element_matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
            element_matches.truncate(3);
            
            // Count matched elements
            if !element_matches.is_empty() {
                matched_elements_count += 1;
            }
            
            // Create element match result with top 3 matches
            let element_result = ElementMatchResult {
                xml_element_id: xml_element.id.clone(),
                xml_text: xml_element.text.clone(),
                top_matches: element_matches,
            };
            
            // Add to overall results
            results.push(element_result);
        }
        
        // Create statistics
        let processed_xml_elements = xml_elements.len().min(50);
        let unmatched_elements = processed_xml_elements.saturating_sub(matched_elements_count);
        
        let statistics = MatchStatistics {
            total_xml_documents: self.xml_documents.len(),
            total_xml_elements: xml_elements.len(),
            total_pdf_blocks: pdf_blocks.len(),
            matched_elements: matched_elements_count,
            unmatched_elements,
            match_threshold: 0.01,
            top_k_matches: self.config.top_k_matches,
        };
        
        (results, statistics)
    }
    
    fn calculate_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simple Jaccard similarity on word sets
        let words1: Vec<&str> = text1.split_whitespace().collect();
        let words2: Vec<&str> = text2.split_whitespace().collect();
        
        let set1: std::collections::HashSet<_> = words1.into_iter().collect();
        let set2: std::collections::HashSet<_> = words2.into_iter().collect();
        
        let intersection: usize = set1.intersection(&set2).count();
        let union: usize = set1.union(&set2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
    
    async fn save_results(&self, pdf_name: &str, document_match: &DocumentMatch) -> Result<()> {
        let output_path = self.config.output_dir.join(format!("{}_matches.json", pdf_name));
        let json_content = serde_json::to_string_pretty(document_match)?;
        fs::write(&output_path, json_content)?;
        info!("Saved matching results to {:?}", output_path);
        
        // Log statistics
        info!("Matching statistics for {}:", pdf_name);
        info!("  Total XML documents processed: {}", document_match.statistics.total_xml_documents);
        info!("  Total XML elements processed: {}", document_match.statistics.total_xml_elements);
        info!("  Total PDF blocks processed: {}", document_match.statistics.total_pdf_blocks);
        info!("  Matched elements: {}", document_match.statistics.matched_elements);
        info!("  Unmatched elements: {}", document_match.statistics.unmatched_elements);
        info!("  Match threshold: {}", document_match.statistics.match_threshold);
        info!("  Top K matches: {}", document_match.statistics.top_k_matches);
        info!("  Best matching XML document: {} (score: {})", 
              document_match.best_match.xml_document_id, document_match.best_match.similarity);
        
        // Log top 5 document similarities
        info!("  Top 5 document similarities:");
        for (i, doc_sim) in document_match.all_document_scores.iter().take(5).enumerate() {
            info!("    {}. {} (score: {})", i + 1, doc_sim.xml_document_id, doc_sim.similarity);
        }
        
        Ok(())
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