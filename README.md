# Enhanced Multi-Lingual Document Matching System

This enhanced document matching system matches translated PDF content against authoritative XML documents using advanced embedding-based similarity techniques combined with structural and lexical features for improved accuracy across all supported languages.

## Enhanced Approach for Multi-Lingual Document Matching

The enhanced system uses a sophisticated multi-level matching approach that combines semantic similarity with structural features for improved accuracy:

### 1. Document-Level Matching

For each PDF document, the system identifies the best matching XML document through an enhanced process:

1. **Extract First 10 Elements**: 
   - From the PDF: Take the first 10 text blocks
   - From each XML document: Take the first 10 elements

2. **Generate Embeddings**:
   - For PDF text blocks: Generate embeddings using the `search_query` prefix
   - For XML elements: Generate embeddings using the `search_document` prefix
   - Both use the nomic-embed-text-v2-moe model via llama.cpp

3. **Calculate Multiple Similarity Metrics**:
   - **Semantic Similarity**: Cosine similarity between embeddings
   - **Length Similarity**: Ratio of smaller character length to larger one
   - **Word Count Similarity**: Ratio of smaller word count to larger one

4. **Compute Composite Scores**:
   - Combine metrics using weighted formula: `Composite = (0.7 × Semantic) + (0.15 × Length) + (0.15 × Word)`
   - For each PDF embedding, find the maximum composite score with any XML embedding
   - Compute average of these 10 maximum composite scores

5. **Enhance with Filename Similarity**:
   - Normalize filenames (replace `-` and `_` with spaces)
   - Compute Jaccard similarity between filename word sets
   - Integrate as secondary factor in final document ranking

6. **Rank Documents**:
   - Each XML document receives a final similarity score
   - Documents are ranked by their enhanced scores
   - The highest-scoring XML document is selected as the best match

### 2. Paragraph-Level Matching

After identifying the best matching XML document, the system performs detailed matching:

1. **Embed All Elements**:
   - Generate embeddings for all PDF blocks and XML elements

2. **Calculate Enhanced Similarities**:
   - For each pair, compute semantic, length, and word count similarities
   - Apply composite scoring formula for more discriminatory matching

3. **Find Best Matches**:
   - For each XML element, find the top K matching PDF blocks
   - Use Qdrant vector database for efficient similarity search
   - Return the top 3 matches for each XML element

4. **Statistics Collection**:
   - Count matches above different similarity thresholds (0.1 and 0.3) for quality assessment

## Technical Details

- **Embedding Model**: nomic-embed-text-v2-moe (768 dimensions, truncated to 256)
- **Similarity Metrics**: 
  - Semantic similarity (cosine similarity of embeddings)
  - Structural similarity (length and word count ratios)
  - Composite scoring (weighted combination of all metrics)
- **Vector Database**: Qdrant for efficient nearest neighbor search
- **Caching**: XML embeddings are cached in `xml_embeddings_cache.json` in the project root

## Output Files

The system generates detailed JSON output files containing:
- Document-level similarity scores for all XML documents (using enhanced composite scoring)
- Paragraph-level matches with enhanced similarity scores (always top 3 per XML element)
- Statistics about the matching process including counts of matches above 0.1 and 0.3 thresholds
- Filename similarity information as a secondary matching factor

## Detailed Technical Documentation

For a comprehensive understanding of the matching process, see:
- [DOCUMENT_MATCHING_PROCESS.md](DOCUMENT_MATCHING_PROCESS.md) for the original approach
- [ENHANCED_MATCHING_APPROACH.md](ENHANCED_MATCHING_APPROACH.md) for the enhanced multi-lingual approach

## Sample Data Collection

To gather sample data used in the matching process, run:
```bash
./gather_samples.sh
```

## Process Example

To see an example of how the enhanced matching process works, run:
```bash
./enhanced_matching_demo.sh
```