#!/bin/bash

# Script to gather text samples used in document matching process

echo "=== Document Matching Process Overview ==="
echo

echo "1. PDF Document Sampling:"
echo "------------------------"
echo "Taking first 10 text blocks from each PDF document:"
echo

# Find a sample PDF processed file
PDF_FILE=$(find pdf_source_inputs -name "*_processed.json" | head -1)

if [ -n "$PDF_FILE" ]; then
    echo "Sample PDF file: $PDF_FILE"
    echo "First 10 text blocks:"
    jq -c '.[0:10][] | {id: .id, text: .text}' "$PDF_FILE" | head -10
else
    echo "No PDF processed files found"
fi

echo
echo "2. XML Document Sampling:"
echo "------------------------"
echo "Taking first 10 elements from each XML document:"
echo

# Find all XML files
XML_FILES=$(find xml_json_inputs -name "*.json")

if [ -n "$XML_FILES" ]; then
    # Process the first few XML files found
    COUNT=0
    for XML_FILE in $XML_FILES; do
        if [ $COUNT -ge 3 ]; then
            break
        fi
        
        echo "Sample XML file: $XML_FILE"
        echo "First 10 elements:"
        jq -c '.[0:10][] | {id: .id, text: .text}' "$XML_FILE" | head -10
        echo
        
        COUNT=$((COUNT + 1))
    done
else
    echo "No XML files found"
fi

# Specifically look for Paris Talks if it exists
PARIS_TALKS=$(find xml_json_inputs -name "*Paris*" -o -name "*paris*" | head -1)
if [ -n "$PARIS_TALKS" ]; then
    echo "Paris Talks XML document:"
    echo "First 10 elements:"
    jq -c '.[0:10][] | {id: .id, text: .text}' "$PARIS_TALKS" | head -10
    echo
fi

echo "3. Embedding Generation:"
echo "-----------------------"
echo "For PDF text blocks: Using 'search_query:' prefix"
echo "For XML elements: Using 'search_document:' prefix"
echo

echo "Example prompts sent to embedding model:"
echo "PDF prompt: search_query: [text content]"
echo "XML prompt: search_document: [text content]"
echo

echo "4. Similarity Calculation:"
echo "-------------------------"
echo "Computing cosine similarity between each PDF embedding and each XML embedding"
echo "Average of all pairwise similarities = Document similarity score"