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

/// Text statistics for similarity calculations
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextStatistics {
    pub char_length: usize,
    pub word_count: usize,
}

impl TextStatistics {
    pub fn new(text: &str) -> Self {
        Self {
            char_length: text.chars().count(),
            word_count: text.split_whitespace().count(),
        }
    }
    
    /// Calculate length similarity as ratio of smaller to larger character length
    pub fn length_similarity(&self, other: &TextStatistics) -> f32 {
        if self.char_length == 0 && other.char_length == 0 {
            1.0
        } else if self.char_length == 0 || other.char_length == 0 {
            0.0
        } else {
            let min_len = self.char_length.min(other.char_length) as f32;
            let max_len = self.char_length.max(other.char_length) as f32;
            min_len / max_len
        }
    }
    
    /// Calculate word count similarity as ratio of smaller to larger word count
    pub fn word_similarity(&self, other: &TextStatistics) -> f32 {
        if self.word_count == 0 && other.word_count == 0 {
            1.0
        } else if self.word_count == 0 || other.word_count == 0 {
            0.0
        } else {
            let min_words = self.word_count.min(other.word_count) as f32;
            let max_words = self.word_count.max(other.word_count) as f32;
            min_words / max_words
        }
    }
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

/// Information about top-scoring phrase pairs for a document
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TopPhrasePair {
    pub xml_element_id: String,
    pub xml_text: String,
    pub pdf_block_id: String,
    pub pdf_text: String,
    pub similarity: f32,
}

/// Document-level match information
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DocumentMatchInfo {
    pub xml_document_id: String,
    pub similarity: f32,
    pub top_phrase_pairs: Vec<TopPhrasePair>,
}

/// Document matching result
#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentMatch {
    pub pdf_document_id: String,
    pub best_match: DocumentMatchInfo,
    pub all_document_scores: Vec<DocumentMatchInfo>,
    pub paragraph_matches: Vec<ElementMatchResult>,
    pub top_phrase_matches: Vec<TopPhraseMatch>,
    pub statistics: MatchStatistics,
}

/// Information about top-scoring phrase pairs
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TopPhraseMatch {
    pub xml_document_id: String,
    pub xml_element_id: String,
    pub xml_text: String,
    pub pdf_block_id: String,
    pub pdf_text: String,
    pub similarity: f32,
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
    pub matches_above_01: usize,
    pub matches_above_03: usize,
    pub top_phrase_matches: Vec<TopPhraseMatch>,
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