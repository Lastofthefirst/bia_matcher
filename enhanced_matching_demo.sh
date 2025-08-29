#!/bin/bash

# Enhanced Document Matching Demo Script

echo "=== Enhanced Multi-Lingual Document Matching Demo ==="
echo

# Check if the matching binary exists
if [ ! -f "./target/release/matching" ] && [ ! -f "./target/debug/matching" ]; then
    echo "Building the matching application..."
    cargo build --release
    if [ $? -ne 0 ]; then
        echo "Failed to build the application"
        exit 1
    fi
fi

# Determine which binary to use
if [ -f "./target/release/matching" ]; then
    MATCHING_BIN="./target/release/matching"
else
    MATCHING_BIN="./target/debug/matching"
fi

echo "Using matching binary: $MATCHING_BIN"
echo

# Show the enhanced approach description
echo "Enhanced Multi-Lingual Document Matching Approach:"
echo "---------------------------------------------------"
echo "1. Semantic Similarity: Uses nomic-embed-text-v2-moe model for embedding generation"
echo "2. Structural Features: Length and word count ratios for language-agnostic signals"
echo "3. Composite Scoring: Weighted combination of semantic and structural similarities"
echo "4. Max Pooling: Focuses on strongest matches while ignoring noise"
echo "5. Filename Enhancement: Considers filename similarities as secondary factor"
echo

# Show sample command
echo "Sample command to run enhanced matching:"
echo "----------------------------------------"
echo "$MATCHING_BIN \\"
echo "  --model-path nomic-embed-text-v2-moe.f16.gguf \\"
echo "  --xml-documents-dir xml_json_inputs \\"
echo "  --pdf-input-dir pdf_source_inputs \\"
echo "  --output-dir output \\"
echo "  --similarity-threshold 0.01 \\"
echo "  --top-k-matches 3"
echo

echo "The enhanced approach combines:"
echo "- 70% semantic similarity (from embeddings)"
echo "- 15% length similarity (structural feature)"
echo "- 15% word count similarity (structural feature)"
echo
echo "Additionally, filename similarity is used as a secondary factor"
echo "to boost scores of documents with similar naming patterns."
echo

echo "This approach provides better discrimination between correct and"
echo "incorrect matches across all supported languages by:"
echo "1. Rewarding matches that are semantically, structurally, and"
echo "   lexically similar"
echo "2. Penalizing matches that only align in one dimension"
echo "3. Widening the score gap between correct and incorrect matches"
echo

echo "To run the full matching process, execute:"
echo "$MATCHING_BIN"