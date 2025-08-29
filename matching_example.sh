#!/bin/bash

# Example script demonstrating the document matching process

echo "=== Document Matching Process Example ==="
echo

echo "Step 1: Sample Extraction"
echo "------------------------"
echo "PDF first 10 blocks:"
echo "  1. 'This is the first paragraph of the translated document.'"
echo "  2. 'The second paragraph contains additional content.'"
echo "  3. 'More text content follows in the third paragraph.'"
echo "  ..."
echo "  10. 'The tenth paragraph concludes our sample.'"
echo

echo "XML first 10 elements:"
echo "  1. 'Tablets of Bahá'u'lláh'"
echo "  2. 'Chapter One: The Declaration'"
echo "  3. 'O My Brother!'"
echo "  ..."
echo "  10. 'Know thou that by 'Brother' is meant every soul'"
echo

echo "Step 2: Embedding Generation"
echo "---------------------------"
echo "PDF embeddings (with search_query: prefix):"
echo "  Embedding 1: [0.12, -0.05, 0.33, ...]"
echo "  Embedding 2: [-0.08, 0.21, 0.15, ...]"
echo "  ..."
echo "  Embedding 10: [0.05, 0.11, -0.09, ...]"
echo

echo "XML embeddings (with search_document: prefix):"
echo "  Embedding 1: [0.10, -0.03, 0.31, ...]"
echo "  Embedding 2: [-0.06, 0.19, 0.17, ...]"
echo "  ..."
echo "  Embedding 10: [0.07, 0.09, -0.11, ...]"
echo

echo "Step 3: Robust Similarity Calculation (Max Pooling)"
echo "---------------------------------------------------"
echo "Computing cosine similarity matrix (10x10):"
echo "               XML_1  XML_2  XML_3  ...  XML_10"
echo "  PDF_1         0.85   0.42   0.21   ...   0.15"
echo "  PDF_2         0.33   0.76   0.65   ...   0.22"
echo "  PDF_3         0.12   0.18   0.88   ...   0.31"
echo "  ..."
echo "  PDF_10        0.25   0.19   0.44   ...   0.76"
echo
echo "For each PDF embedding, find the maximum similarity:"
echo "  Max similarity for PDF_1: 0.85 (matches XML_1)"
echo "  Max similarity for PDF_2: 0.76 (matches XML_2)"
echo "  Max similarity for PDF_3: 0.88 (matches XML_3)"
echo "  ..."
echo "  Max similarity for PDF_10: 0.76 (matches XML_10)"
echo

echo "Step 4: Average of Maximum Similarities"
echo "--------------------------------------"
echo "Average of max similarities = (0.85 + 0.76 + 0.88 + ... + 0.76) / 10 = 0.724"
echo

echo "Step 5: Document Ranking"
echo "-----------------------"
echo "Comparing with other XML documents:"
echo "  Tablets of Bahá'u'lláh: 0.724 (best match)"
echo "  Kitáb-i-Aqdas: 0.512"
echo "  The Hidden Words: 0.487"
echo "  ..."
echo

echo "Result: 'Tablets of Bahá'u'lláh' is selected as the best matching document"
echo
echo "Why Max Pooling Works Better:"
echo "----------------------------"
echo "1. Focuses on strongest matches rather than averaging all comparisons"
echo "2. Ignores noise from mismatched content (headers, footers, etc.)"
echo "3. More resilient to structural differences between documents"
echo "4. Amplifies confident matches while minimizing dilution from weak ones"