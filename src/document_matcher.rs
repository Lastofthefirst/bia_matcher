use crate::error::MatchingError;
use crate::models::{Config, DocumentMatch, PdfBlock, XmlElement, ElementMatchResult, MatchResult, DocumentMatchInfo, MatchStatistics};
use crate::embedding_service::EmbeddingService;
use crate::similarity_search::{DocumentSimilaritySearch, ParagraphSimilaritySearch};
use anyhow::Result;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::info;

pub struct DocumentMatcher {
    config: Config,
    xml_documents: HashMap<String, Vec<XmlElement>>,
    pdf_documents: HashMap<String, Vec<PdfBlock>>,
    embedding_service: EmbeddingService,
    document_search: DocumentSimilaritySearch,
}

impl DocumentMatcher {
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing DocumentMatcher with embedding-based matching");
        
        // Initialize embedding service
        let embedding_service = EmbeddingService::new(&config.model_path).await?;
        let embedding_dim = embedding_service.embedding_dim();
        
        // Initialize document similarity search
        let document_search = DocumentSimilaritySearch::new(embedding_dim)?;
        
        Ok(DocumentMatcher {
            config,
            xml_documents: HashMap::new(),
            pdf_documents: HashMap::new(),
            embedding_service,
            document_search,
        })
    }
    
    pub async fn process_all_pdfs(&mut self) -> Result<()> {
        info!("Loading XML documents");
        self.load_xml_documents().await?;
        
        info!("Loading PDF documents");
        self.load_pdf_documents().await?;
        
        info!("Processing PDF documents for matching");
        self.match_documents().await?;
        
        Ok(())
    }
    
    async fn load_xml_documents(&mut self) -> Result<()> {
        let xml_dir = self.config.xml_documents_dir.clone();
        if !xml_dir.exists() {
            return Err(MatchingError::InvalidConfig(format!("XML documents directory does not exist: {:?}", xml_dir)).into());
        }
        
        self.load_xml_documents_recursive(&xml_dir)?;
        
        // Generate document-level embeddings for all XML documents
        for (document_id, elements) in &self.xml_documents {
            // Concatenate first 30 elements or all elements if fewer
            let document_text = elements
                .iter()
                .take(30)
                .map(|e| e.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            
            let embedding = self.embedding_service.generate_document_embedding(&document_text)?;
            self.document_search.add_document(document_id.clone(), embedding)?;
        }
        
        info!("Loaded {} XML documents with embeddings", self.xml_documents.len());
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
        
        // Collect pdf documents to avoid borrowing issues
        let pdf_documents: Vec<(String, Vec<PdfBlock>)> = self.pdf_documents
            .iter()
            .map(|(name, blocks)| (name.clone(), blocks.clone()))
            .collect();
        
        for (pdf_name, pdf_blocks) in pdf_documents {
            info!("Matching PDF document: {}", pdf_name);
            
            // Calculate document-level similarities using embeddings
            let document_similarities = self.calculate_document_similarities_embeddings(&pdf_blocks).await?;
            
            // Find the best matching XML document
            let best_match = document_similarities
                .iter()
                .max_by(|a, b| a.similarity.partial_cmp(&b.similarity).unwrap())
                .unwrap(); // Safe because we checked for empty xml_documents earlier
            
            info!("Best matching XML document for {}: {} (score: {})", 
                  pdf_name, best_match.xml_document_id, best_match.similarity);
            
            // Perform paragraph-level matching with the best XML document using embeddings
            let best_xml_elements = self.xml_documents.get(&best_match.xml_document_id).unwrap().clone();
            let (matches, statistics) = self.embedding_match_paragraphs(&best_xml_elements, &pdf_blocks).await?;
            
            let document_match = DocumentMatch {
                pdf_document_id: pdf_name.clone(),
                best_match: (*best_match).clone(),
                all_document_scores: document_similarities,
                paragraph_matches: matches,
                statistics,
            };
            
            // Save results
            self.save_results(&pdf_name, &document_match).await?;
        }
        
        Ok(())
    }
    
    async fn calculate_document_similarities_embeddings(&mut self, pdf_blocks: &[PdfBlock]) -> Result<Vec<DocumentMatchInfo>> {
        // Get first 30 blocks for document-level matching (as per specification)
        let first_blocks: Vec<&PdfBlock> = pdf_blocks.iter().take(30).collect();
        
        // Concatenate text from first blocks
        let concatenated_text: String = first_blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Generate query embedding for the PDF content
        let query_embedding = self.embedding_service.generate_query_embedding(&concatenated_text)?;
        
        // Get similarities for all documents using FAISS
        let similarities = self.document_search.get_all_document_similarities(&query_embedding)?;
        
        // Convert to DocumentMatchInfo
        let document_similarities = similarities
            .into_iter()
            .map(|(xml_document_id, similarity)| DocumentMatchInfo {
                xml_document_id,
                similarity,
            })
            .collect();
        
        Ok(document_similarities)
    }
    
    async fn embedding_match_paragraphs(&mut self, xml_elements: &[XmlElement], pdf_blocks: &[PdfBlock]) -> Result<(Vec<ElementMatchResult>, MatchStatistics)> {
        let embedding_dim = self.embedding_service.embedding_dim();
        let mut paragraph_search = ParagraphSimilaritySearch::new(embedding_dim)?;
        
        // Generate embeddings for XML elements
        let mut xml_embeddings = Vec::new();
        for xml_element in xml_elements.iter().take(50) {  // Process up to 50 XML elements
            let embedding = self.embedding_service.generate_document_embedding(&xml_element.text)?;
            xml_embeddings.push((xml_element.id.clone(), embedding));
        }
        
        // Add XML element embeddings to the search index
        paragraph_search.add_elements(&xml_embeddings)?;
        
        let mut results = Vec::new();
        let mut matched_elements_count = 0;
        
        // For each XML element, find the top matching PDF blocks
        for xml_element in xml_elements.iter().take(50) {
            let mut element_matches = Vec::new();
            
            // Generate embeddings for all PDF blocks and find matches
            for pdf_block in pdf_blocks.iter() {
                let pdf_embedding = self.embedding_service.generate_query_embedding(&pdf_block.text)?;
                let matches = paragraph_search.find_matches(&pdf_embedding, 1, self.config.similarity_threshold)?;
                
                // Check if this XML element matches
                if let Some((matched_id, similarity)) = matches.first() {
                    if matched_id == &xml_element.id {
                        element_matches.push(MatchResult {
                            pdf_block_id: pdf_block.id.clone(),
                            similarity: *similarity,
                            pdf_text: pdf_block.text.clone(),
                        });
                    }
                }
            }
            
            // Sort by similarity (highest first) and take top 3
            element_matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
            element_matches.truncate(self.config.top_k_matches);
            
            // Count matched elements
            if !element_matches.is_empty() {
                matched_elements_count += 1;
            }
            
            // Create element match result with top matches
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
            match_threshold: self.config.similarity_threshold,
            top_k_matches: self.config.top_k_matches,
        };
        
        Ok((results, statistics))
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