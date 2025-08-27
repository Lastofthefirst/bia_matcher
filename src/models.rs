use serde::{Deserialize, Serialize};

/// PDF block representation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PdfBlock {
    pub id: String,
    #[serde(rename = "block_type")]
    pub block_type: String,
    pub html: String,
    pub text: String,
}

/// XML element representation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct XmlElement {
    pub id: String,
    pub text: String,
    #[serde(rename = "inner_xml")]
    pub inner_xml: String,
}

/// Matching result for a single XML element
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ElementMatchResult {
    pub xml_element_id: String,
    pub xml_text: String,
    pub top_matches: Vec<MatchResult>,
}

/// Individual match result
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MatchResult {
    pub pdf_block_id: String,
    pub similarity: f32,
    pub pdf_text: String,
}

/// Matching result
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MatchResultOld {
    pub xml_element_id: String,
    pub pdf_block_id: String,
    pub similarity: f32,
    pub xml_text: String,
    pub pdf_text: String,
}

/// Document-level match information
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DocumentMatchInfo {
    pub xml_document_id: String,
    pub similarity: f32,
}

/// Document matching result
#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentMatch {
    pub pdf_document_id: String,
    pub best_match: DocumentMatchInfo,
    pub all_document_scores: Vec<DocumentMatchInfo>,
    pub paragraph_matches: Vec<ElementMatchResult>,
    pub statistics: MatchStatistics,
}

/// Matching statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct MatchStatistics {
    pub total_xml_documents: usize,
    pub total_xml_elements: usize,
    pub total_pdf_blocks: usize,
    pub matched_elements: usize,
    pub unmatched_elements: usize,
    pub match_threshold: f32,
    pub top_k_matches: usize,
}

/// Configuration structure
#[derive(Debug, Clone)]
pub struct Config {
    pub model_path: std::path::PathBuf,
    pub xml_documents_dir: std::path::PathBuf,
    pub pdf_input_dir: std::path::PathBuf,
    pub output_dir: std::path::PathBuf,
    pub similarity_threshold: f32,
    pub top_k_matches: usize,
}