# Enhanced Multi-Lingual Document Matching Approach

## Overview

This document describes the enhanced multi-lingual document matching approach that improves the accuracy and reliability of matching translated PDF content against authoritative XML documents. The approach combines semantic similarity with structural features to create a more discriminatory composite score.

## Key Features

### 1. Semantic Similarity
- Uses advanced embedding-based techniques to compute cosine similarity between document sections
- Employs the nomic-embed-text-v2-moe model via llama.cpp for generating high-quality embeddings
- Implements max pooling to focus on the strongest matches while ignoring noise

### 2. Structural Similarity Features
- **Length Similarity**: Ratio of smaller character length to larger one
- **Word Count Similarity**: Ratio of smaller word count to larger one

### 3. Filename Similarity
- Normalizes filenames by replacing hyphens and underscores with spaces
- Computes Jaccard similarity between filename word sets

### 4. Composite Scoring
- Combines semantic, length, and word count similarities using weighted averages
- Formula: `Composite Score = (0.7 × Semantic) + (0.15 × Length) + (0.15 × Word)`
- Integrates filename similarity as a secondary factor

## Implementation Details

### Text Statistics
The system calculates text statistics for each document section:
```rust
pub struct TextStatistics {
    pub char_length: usize,
    pub word_count: usize,
}
```

### Similarity Calculations
1. **Length Similarity**: `min_length / max_length`
2. **Word Similarity**: `min_words / max_words`
3. **Cosine Similarity**: Standard vector dot product calculation
4. **Filename Similarity**: Jaccard similarity of normalized filename words

### Composite Score Calculation
The composite score combines multiple similarity metrics:
```rust
fn calculate_composite_score(semantic: f32, length: f32, word: f32) -> f32 {
    0.7 * semantic + 0.15 * length + 0.15 * word
}
```

### Max Pooling Approach
For each PDF section, the system:
1. Computes similarity with all XML sections
2. Finds the maximum composite score
3. Averages these maximum scores for document-level matching

### Filename Enhancement
The system enhances matching by considering filename similarities:
1. Normalizes filenames (replaces `-` and `_` with spaces)
2. Computes Jaccard similarity between filename word sets
3. Integrates this as a weighted factor in final scoring

## Benefits

### Language-Agnostic Features
- Length and word count ratios provide strong signals independent of language
- Works equally well for all supported languages

### Improved Discrimination
- Correct matches benefit from high scores in all categories
- Incorrect matches are penalized by low structural similarity
- Widens the score gap between correct and incorrect matches

### Relative Ranking
- Ranks documents based on relative scores rather than absolute values
- Ensures consistent performance across different languages

## Expected Outcomes

The enhanced approach should result in:
1. More accurate document-level matching
2. Better paragraph-level alignment
3. Consistent performance across all supported languages
4. Improved discrimination between correct and incorrect matches

## Future Enhancements

Potential areas for further improvement:
1. Dynamic weight adjustment based on language characteristics
2. Additional structural features (sentence count, paragraph structure)
3. Machine learning-based scoring optimization
4. Cross-validation with human-verified matches