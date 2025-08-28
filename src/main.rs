mod document_matcher;
mod error;
mod models;
mod embedding_service;
mod similarity_search;

use anyhow::Result;
use clap::Parser;
use document_matcher::DocumentMatcher;
use models::Config;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the embedding model
    #[clap(long, default_value = "nomic-embed-text-v2-moe.f16.gguf")]
    model_path: PathBuf,

    /// Directory containing XML documents
    #[clap(long, default_value = "xml_json_inputs")]
    xml_documents_dir: PathBuf,

    /// Directory containing PDF input files
    #[clap(long, default_value = "pdf_source_inputs")]
    pdf_input_dir: PathBuf,

    /// Output directory for results
    #[clap(long, default_value = "output")]
    output_dir: PathBuf,

    /// Similarity threshold for matches
    #[clap(long, default_value_t = 0.7)]
    similarity_threshold: f32,

    /// Number of top matches to return
    #[clap(long, default_value_t = 3)]
    top_k_matches: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    let config = Config {
        model_path: args.model_path,
        xml_documents_dir: args.xml_documents_dir,
        pdf_input_dir: args.pdf_input_dir,
        output_dir: args.output_dir,
        similarity_threshold: args.similarity_threshold,
        top_k_matches: args.top_k_matches,
    };
    
    // Create output directory if it doesn't exist
    fs::create_dir_all(&config.output_dir)?;
    
    // Initialize the document matching system
    let mut matcher = DocumentMatcher::new(config).await?;
    
    // Process all PDF files
    matcher.process_all_pdfs().await?;
    
    Ok(())
}
