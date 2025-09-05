# Document Matching System Specification

## Overview

Build a high-performance document matching system in Rust that matches translated PDF content against authoritative XML documents. The system will use FAISS for similarity search and llama.cpp for generating text embeddings with the nomic-embed-text-v2-moe model.

## Input Data Structures

### PDF Input Format (Translation Documents)
```json
[
  {
    "id": "/page/0/Text/1",
    "block_type": "Text",
    "html": "<p>Document content here</p>",
    "text": "Document content here"
  }
]
```

### XML-Associated JSON Format (Reference Documents)
```json
[
  {
    "id": "030537471",
    "text": "Tablets of Bahá'u'lláh",
    "inner_xml": "Tablets of Bahá'u'lláh"
  }
]
```

## Core Components

### 1. Embedding Generation Service
- Use `llama-cpp-rs` to interface with the nomic-embed-text-v2-moe model
- Implement proper prefix handling:
  - `search_document:` for XML content
  - `search_query:` for PDF text blocks
- Support Matryoshka embeddings (truncate to 256 dimensions)
- Implement caching to avoid recomputing embeddings

### 2. Document Matching Engine
- Use `faiss-rs` for similarity search
- Implement two-level matching:
  1. Document-level matching: Compare first 30 chunks of PDF with XML documents
  2. Paragraph-level matching: Find best matches for each XML element

### 3. Data Processing Pipeline
- Process PDF JSON files to extract text blocks
- Process XML-associated JSON files to extract elements with IDs
- Handle text normalization and cleaning

## Implementation Details

### Data Structures

```rust
// PDF block representation
#[derive(Debug, Serialize, Deserialize, Clone)]
struct PdfBlock {
    id: String,
    block_type: String,
    html: String,
    text: String,
}

// XML element representation
#[derive(Debug, Serialize, Deserialize, Clone)]
struct XmlElement {
    id: String,
    text: String,
    inner_xml: String,
}

// Matching result
#[derive(Debug, Serialize, Deserialize)]
struct MatchResult {
    xml_element_id: String,
    pdf_block_id: String,
    similarity: f32,
    xml_text: String,
    pdf_text: String,
}

// Document matching result
#[derive(Debug, Serialize, Deserialize)]
struct DocumentMatch {
    xml_document_id: String,
    similarity: f32,
    paragraph_matches: Vec<MatchResult>,
}
```

### Main Processing Flow

1. **Initialization**
   - Load and preprocess all XML reference documents
   - Generate and store embeddings for XML elements
   - Build FAISS indices for document and paragraph matching

2. **PDF Processing**
   - For each PDF JSON file:
     - Extract text blocks
     - Generate embeddings for first 30 blocks (document matching)
     - Find best-matching XML document using FAISS

3. **Paragraph Matching**
   - For the matched XML document:
     - Generate embeddings for all PDF blocks
     - Find best matches for each XML element using FAISS
     - Return top 3 matches for each XML element

4. **Output Generation**
   - Create structured output with matching results
   - Include similarity scores and source texts
   - Format for easy integration with XML population

### Performance Optimization

- Use async processing for embedding generation
- Implement batch processing for efficiency
- Use memory-mapped storage for FAISS indices
- Implement caching for frequently accessed data

### Error Handling

- Comprehensive error handling for:
  - File I/O operations
  - Embedding generation failures
  - FAISS index errors
  - Data parsing issues

### Configuration

```rust
struct Config {
    model_path: PathBuf,
    xml_documents_dir: PathBuf,
    pdf_input_dir: PathBuf,
    output_dir: PathBuf,
    similarity_threshold: f32,
    top_k_matches: usize,
}
```

## Key Algorithms

### Document Matching
1. Concatenate first 30 text blocks from PDF
2. Generate embedding with `search_query:` prefix
3. Compare against XML document embeddings (generated with `search_document:` prefix)
4. Return best match with similarity score

### Paragraph Matching
1. For each XML element, generate embedding with `search_document:` prefix
2. For each PDF block, generate embedding with `search_query:` prefix
3. Use FAISS to find top K matches for each XML element
4. Apply similarity threshold to filter weak matches

## Integration Points

### Input
- Directory of PDF JSON files (translations)
- Directory of XML-associated JSON files (reference documents)

### Output
- JSON files containing matching results
- Structured data for XML population
- Similarity scores for quality assessment

### External Services
- llama.cpp server for embedding generation
- FAISS for similarity search

## Implementation Considerations

1. **Memory Management**
   - Process documents in chunks to avoid excessive memory usage
   - Use memory-mapped files for large datasets
   - Implement careful ownership and borrowing patterns

2. **Performance**
   - Use Rayon for parallel processing
   - Implement efficient batch operations
   - Optimize FA index building and querying

3. **Accuracy**
   - Implement proper text normalization
   - Handle multilingual content appropriately
   - Tune similarity thresholds based on validation

4. **Extensibility**
   - Design modular components for easy maintenance
   - Support additional embedding models
   - Allow configuration of matching strategies

## Testing Strategy

1. Unit tests for individual components
2. Integration tests for full pipeline
3. Validation against known matching pairs
4. Performance benchmarking
5. Accuracy measurement against golden dataset

## Deployment

- Standalone Rust binary
- Configuration via file or command line
- Minimal external dependencies
- Efficient resource usage

This specification provides a comprehensive guide for implementing a high-performance document matching system in Rust that meets the requirements for matching translated content against authoritative XML documents.





marker_single \
  --llm_service marker.services.openai.OpenAIService \
  --openai_base_url https://openrouter.ai/api/v1 \
--openai_api_key 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 \
  --openai_model deepseek/deepseek-r1:free \
  --use_llm \
  FPATH

    