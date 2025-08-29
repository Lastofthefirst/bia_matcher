# Document Matching Process - Technical Overview

## Overview

This document provides a detailed explanation of the document matching process used to match translated PDF content against authoritative XML documents. The system uses embedding-based similarity techniques to identify the best matching XML document for each PDF.

## Process Flow

### 1. Document-Level Matching

The document-level matching process identifies which XML document best matches a given PDF document by comparing their first 10 elements using a max pooling approach to improve accuracy.

#### Step 1: Sample Extraction
- From each PDF: Extract the first 10 text blocks
- From each XML document: Extract the first 10 elements

#### Step 2: Embedding Generation
- For PDF text blocks: Generate embeddings with `search_query:` prefix
- For XML elements: Generate embeddings with `search_document:` prefix
- Uses nomic-embed-text-v2-moe model via llama.cpp
- Embeddings are 768-dimensional, truncated to 256 dimensions

#### Step 3: Robust Similarity Calculation (Max Pooling)
For each XML document:
1. Compare each of the 10 PDF embeddings with each of the 10 XML embeddings
2. Calculate cosine similarity between each pair:
   ```
   cosine_similarity = (A · B) / (||A|| × ||B||)
   ```
   where A and B are embedding vectors
3. For each PDF embedding, find the maximum similarity with any XML embedding (max pooling)
4. Compute average of these 10 maximum similarity scores

This max pooling approach focuses on the strongest matches while ignoring noise from mismatched content, making it more robust for translation matching.

#### Step 4: Document Ranking
- Sort XML documents by their average maximum similarity scores (highest first)
- Select the XML document with the highest score as the best match

### 2. Paragraph-Level Matching

After identifying the best matching XML document, detailed matching is performed:

#### Step 1: Full Embedding Generation
- Generate embeddings for all PDF blocks and XML elements

#### Step 2: Similarity Search
- For each XML element, find the top K matching PDF blocks (always returns top 3, ignoring similarity threshold)
- Use Qdrant vector database for efficient nearest neighbor search

#### Step 3: Statistics Collection
- Count matches above different similarity thresholds (0.1 and 0.3) for quality assessment

## Code Implementation

### Key Functions

#### `calculate_document_similarities()`
Located in `src/document_matcher.rs`, this function performs document-level matching using the max pooling approach:

```rust
async fn calculate_document_similarities(&self, pdf_blocks: &[PdfBlock]) -> Result<Vec<DocumentMatchInfo>> {
    // Take first 10 blocks
    let first_10_pdf_texts: Vec<&str> = pdf_blocks.iter().take(10).map(|b| b.text.as_str()).collect();
    
    // Generate query embeddings for PDF
    let pdf_embeddings = self.embedder.generate_query_embeddings(&first_10_pdf_texts).await?;
    
    // Truncate to 256 dimensions
    let truncated_pdf_embeddings = pdf_embeddings.slice_move(ndarray::s![.., ..256]);
    
    let mut similarities = Vec::new();
    
    // For each XML document
    for (doc_id, xml_elements) in &self.xml_documents {
        // Take first 10 elements
        let first_10_xml_texts: Vec<&str> = xml_elements.iter().take(10).map(|e| e.text.as_str()).collect();
        
        // Generate document embeddings for XML
        let xml_embeddings = self.embedder.generate_embeddings(&first_10_xml_texts).await?;
        
        // Truncate to 256 dimensions
        let truncated_xml_embeddings = xml_embeddings.slice_move(ndarray::s![.., ..256]);
        
        // Calculate max similarity for each PDF embedding (max pooling approach)
        let mut max_similarities = Vec::new();
        
        // For each PDF embedding, find the maximum similarity with any XML embedding
        for pdf_row in truncated_pdf_embeddings.rows() {
            let pdf_vector: Vec<f32> = pdf_row.to_vec();
            
            // Skip zero vectors
            if pdf_vector.iter().all(|&x| x == 0.0) {
                max_similarities.push(0.0);
                continue;
            }
            
            let mut max_similarity = f32::NEG_INFINITY;
            
            // Find the maximum similarity with any XML embedding
            for xml_row in truncated_xml_embeddings.rows() {
                let xml_vector: Vec<f32> = xml_row.to_vec();
                
                // Skip zero vectors
                if xml_vector.iter().all(|&x| x == 0.0) {
                    continue;
                }
                
                // Calculate cosine similarity
                let dot_product: f32 = pdf_vector.iter().zip(xml_vector.iter()).map(|(a, b)| a * b).sum();
                let pdf_magnitude: f32 = pdf_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                let xml_magnitude: f32 = xml_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                
                if pdf_magnitude > 0.0 && xml_magnitude > 0.0 {
                    let similarity = dot_product / (pdf_magnitude * xml_magnitude);
                    if similarity > max_similarity {
                        max_similarity = similarity;
                    }
                }
            }
            
            // If no valid similarity was found, use 0.0
            if max_similarity == f32::NEG_INFINITY {
                max_similarity = 0.0;
            }
            
            max_similarities.push(max_similarity);
        }
        
        // Calculate average of maximum similarities (max pooling approach)
        let avg_max_similarity = if !max_similarities.is_empty() {
            max_similarities.iter().sum::<f32>() / max_similarities.len() as f32
        } else {
            0.0
        };
        
        similarities.push(DocumentMatchInfo {
            xml_document_id: doc_id.clone(),
            similarity: avg_max_similarity,
        });
    }
    
    // Sort by similarity (highest first)
    similarities.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(similarities)
}
```

### Sample Data Format

#### PDF Blocks
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

#### XML Elements
```json
[
  {
    "id": "030537471",
    "text": "Tablets of Bahá'u'lláh",
    "inner_xml": "Tablets of Bahá'u'lláh"
  }
]
```

## Performance Considerations

1. **Caching**: XML embeddings are cached in `xml_embeddings_cache.json` to avoid recomputation
2. **Batching**: Embedding generation processes texts in batches of 10 for efficiency
3. **Truncation**: Embeddings are truncated to 256 dimensions for faster processing
4. **Vector Database**: Qdrant is used for efficient similarity search in paragraph-level matching

## Configuration

The system can be configured with the following parameters:
- `similarity_threshold`: Minimum similarity score for matches (default: 0.01) - Note: Currently ignored in paragraph matching
- `top_k_matches`: Number of top matches to return (default: 3)
- `clear_cache`: Option to clear embedding cache before processing

## Output Format

The system generates detailed JSON output files containing:
- Document-level similarity scores for all XML documents
- Paragraph-level matches with similarity scores (always top 3 per XML element)
- Statistics about the matching process including counts of matches above 0.1 and 0.3 thresholds