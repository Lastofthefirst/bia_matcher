# Document Matching System

This is a high-performance document matching system implemented in Rust that matches translated PDF content against authoritative XML documents.

## Features

- Matches translated PDF content against authoritative XML documents
- Uses text similarity algorithms to find matching sections
- Provides detailed statistics about the matching process
- Recursively traverses subdirectories to find documents
- Calculates document-level similarity scores for each XML document
- Processes JSON representations of both PDF and XML documents
- Outputs structured matching results in JSON format

## Input Data Formats

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

## Getting Started

### Prerequisites

- Rust toolchain (https://www.rust-lang.org/tools/install)

### Building

```bash
cargo build
```

### Running

```bash
cargo run [OPTIONS]
```

### Command Line Options

- `--model-path`: Path to the embedding model (default: nomic-embed-text-v2-moe.f16.gguf)
- `--xml-documents-dir`: Directory containing XML documents (default: xml_json_inputs)
- `--pdf-input-dir`: Directory containing PDF input files (default: pdf_source_inputs)
- `--output-dir`: Output directory for results (default: output)
- `--similarity-threshold`: Similarity threshold for matches (default: 0.7)
- `--top-k-matches`: Number of top matches to return (default: 3)

## Project Structure

- `src/main.rs`: Entry point of the application
- `src/document_matcher.rs`: Core document matching logic
- `src/models.rs`: Data structures used in the application
- `src/error.rs`: Custom error types
- `pdf_source_inputs/`: Directory containing PDF JSON files (recursively searched)
- `xml_json_inputs/`: Directory containing XML-associated JSON files (recursively searched)
- `output/`: Directory where matching results are saved

## How It Works

1. The application recursively loads all XML documents from the specified directory and its subdirectories
2. It recursively loads all PDF documents (files ending with "_processed.json") from the specified directory and its subdirectories
3. For each PDF document:
   - Calculates document-level similarity scores against all XML documents
   - Identifies the best matching XML document
   - Performs paragraph-level matching between the PDF and the best matching XML document
4. Results are saved as JSON files in the output directory

## Output Format

The system generates detailed output that includes:

1. **Document-Level Matching**:
   - Scores for each XML document compared to the input PDF
   - Identification of the best matching XML document

2. **Paragraph-Level Matching**:
   - List of specific matches between XML elements and PDF blocks
   - Similarity scores for each match (now generating over 60 matches per document)

3. **Statistics**: Comprehensive information about the matching process:
   - Total number of XML documents processed
   - Total number of XML elements processed
   - Total number of PDF blocks processed
   - Number of matched elements
   - Number of unmatched elements
   - Match threshold used
   - Top-K matches configuration

## Algorithm

The current implementation uses a simple Jaccard similarity algorithm based on word overlap between text segments. In a production environment, this would be replaced with more sophisticated embedding-based matching using models like nomic-embed-text-v2-moe.

Praise God