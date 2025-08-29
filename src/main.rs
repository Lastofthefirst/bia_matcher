mod document_matcher;
mod error;
mod models;

use anyhow::Result;
use clap::Parser;
use document_matcher::DocumentMatcher;
use models::Config;
use std::fs;
use std::path::PathBuf;
use tracing::info;

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
    #[clap(long, default_value_t = 0.01)]
    similarity_threshold: f32,

    /// Number of top matches to return
    #[clap(long, default_value_t = 3)]
    top_k_matches: usize,
    
    /// Clear the embedding cache before processing
    #[clap(long)]
    clear_cache: bool,
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
    
    // Clear cache if requested
    if args.clear_cache {
        let cache_path = config.output_dir.join("xml_embeddings_cache.json");
        if cache_path.exists() {
            fs::remove_file(&cache_path)?;
            info!("Cleared embedding cache");
        }
    }
    
    // Initialize the document matching system
    let mut matcher = DocumentMatcher::new(config).await?;
    
    // Process all PDF files
    info!("Starting PDF processing with enhanced multi-lingual document matching");
    matcher.process_all_pdfs().await?;
    info!("PDF processing completed");
    
    // Stop Qdrant container
    info!("Stopping Qdrant container");
    matcher.stop_qdrant_container().await?;
    info!("Qdrant container stopped");
    
    // Explicitly drop the matcher to ensure all resources are released
    info!("Dropping matcher");
    drop(matcher);
    info!("Matcher dropped");
    
    // Force cleanup of any remaining async runtime resources
    info!("Forcing cleanup of async runtime");
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    info!("Async runtime cleanup completed");
    
    // Check if there are any remaining Docker containers
    info!("Checking for remaining Docker containers...");
    let docker_check = tokio::task::spawn_blocking(|| {
        std::process::Command::new("docker")
            .arg("ps")
            .arg("--filter")
            .arg("name=qdrant-matching-app")
            .output()
    }).await??;
    
    if docker_check.status.success() {
        let output = String::from_utf8_lossy(&docker_check.stdout);
        if output.contains("qdrant-matching-app") {
            info!("Warning: Qdrant container still appears to be running");
        } else {
            info!("No Qdrant containers found running");
        }
    }
    
    // Check for any remaining llama-server processes
    info!("Checking for remaining llama-server processes...");
    let llama_check = tokio::task::spawn_blocking(|| {
        std::process::Command::new("pgrep")
            .arg("llama-server")
            .output()
    }).await??;
    
    if llama_check.status.success() && !llama_check.stdout.is_empty() {
        info!("Warning: llama-server processes still appear to be running");
    } else {
        info!("No llama-server processes found running");
    }
    
    info!("Program completed successfully with enhanced multi-lingual document matching");
    Ok(())
}
