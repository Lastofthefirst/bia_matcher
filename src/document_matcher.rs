use crate::error::MatchingError;
use crate::models::{Config, DocumentMatch, PdfBlock, XmlElement, ElementMatchResult, MatchResult, DocumentMatchInfo, MatchStatistics, TopPhraseMatch, TopPhrasePair, TextStatistics};
use anyhow::Result;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use tracing::info;
use reqwest;
use ndarray::{Array, Array2};
use qdrant_client::qdrant::{PointStruct, VectorParamsBuilder, Distance, CreateCollectionBuilder, UpsertPointsBuilder, QueryPointsBuilder};
use qdrant_client::Qdrant;

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        0.0
    } else {
        dot_product / (magnitude_a * magnitude_b)
    }
}

/// Calculate composite score using weighted combination of similarities
/// Formula: Composite Score = (0.7 * Semantic) + (0.15 * Length) + (0.15 * Word)
fn calculate_composite_score(semantic: f32, length: f32, word: f32) -> f32 {
    0.7 * semantic + 0.15 * length + 0.15 * word
}

/// Normalize filename by replacing - and _ with spaces
fn normalize_filename(filename: &str) -> String {
    filename.replace("-", " ").replace("_", " ")
}

/// Calculate similarity between two filenames
fn filename_similarity(filename1: &str, filename2: &str) -> f32 {
    let norm1 = normalize_filename(filename1);
    let norm2 = normalize_filename(filename2);
    
    // Simple Jaccard similarity for words
    let words1: Vec<&str> = norm1.split_whitespace().collect();
    let words2: Vec<&str> = norm2.split_whitespace().collect();
    
    if words1.is_empty() && words2.is_empty() {
        return 1.0;
    }
    
    let set1: std::collections::HashSet<&str> = words1.into_iter().collect();
    let set2: std::collections::HashSet<&str> = words2.into_iter().collect();
    
    let intersection: usize = set1.intersection(&set2).count();
    let union: usize = set1.union(&set2).count();
    
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

pub struct EmbeddingGenerator {
    model_path: String,
    server_port: u16,
    client: reqwest::Client,
    child_process: Option<std::process::Child>,
}

impl EmbeddingGenerator {
    pub fn new(model_path: String) -> Result<Self> {
        info!("Initializing embedding generator with model: {}", model_path);
        let server_port = 8080; // Default port
        
        Ok(Self {
            model_path,
            server_port,
            client: reqwest::Client::new(),
            child_process: None,
        })
    }

    pub async fn start_server(&mut self) -> Result<()> {
        info!("Starting llama.cpp server...");
        
        // Start the llama.cpp server in the background
        let child = Command::new("llama-server")
            .arg("--model")
            .arg(&self.model_path)
            .arg("--port")
            .arg(self.server_port.to_string())
            .arg("--embedding")
            .arg("--host")
            .arg("127.0.0.1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
            
        self.child_process = Some(child);
            
        // Give the server some time to start
        thread::sleep(Duration::from_secs(3));
        
        // Check if the server is responding
        let max_retries = 10;
        for i in 0..max_retries {
            match self.client.get(&format!("http://127.0.0.1:{}/health", self.server_port)).send().await {
                Ok(response) if response.status().is_success() => {
                    info!("llama.cpp server started successfully");
                    return Ok(());
                }
                _ => {
                    if i == max_retries - 1 {
                        return Err(anyhow::anyhow!("Failed to start llama.cpp server"));
                    }
                    thread::sleep(Duration::from_millis(500));
                }
            }
        }
        
        Ok(())
    }

    pub async fn stop_server(&mut self) -> Result<()> {
        info!("Stopping llama.cpp server...");
        
        // Try to gracefully shutdown the server
        info!("Sending shutdown request to llama.cpp server...");
        match tokio::time::timeout(
            Duration::from_secs(5),
            self.client.post(&format!("http://127.0.0.1:{}/shutdown", self.server_port))
                .send()
        ).await {
            Ok(Ok(response)) => {
                info!("Shutdown request sent, status: {}", response.status());
            }
            Ok(Err(e)) => {
                info!("Failed to send shutdown request: {}", e);
            }
            Err(_) => {
                info!("Shutdown request timed out");
            }
        }
            
        // Give it a moment to shutdown gracefully
        info!("Waiting for graceful shutdown...");
        tokio::time::sleep(Duration::from_secs(2)).await;
        info!("Graceful shutdown wait completed");
        
        // If we have a child process, try to terminate it properly
        if let Some(mut child) = self.child_process.take() {
            info!("Terminating child process");
            
            // Try to terminate the process gracefully
            match child.kill() {
                Ok(_) => {
                    info!("Sent SIGKILL to child process");
                }
                Err(e) => {
                    info!("Failed to kill child process: {}", e);
                }
            }
            
            // Wait for the process to exit
            info!("Waiting for child process to exit...");
            match child.wait() {
                Ok(status) => {
                    info!("Child process exited with status: {:?}", status);
                }
                Err(e) => {
                    info!("Failed to wait for child process: {}", e);
                }
            }
            info!("Child process termination completed");
        }
        
        // Force kill any remaining llama-server processes
        info!("Killing any remaining llama-server processes...");
        match Command::new("pkill")
            .arg("-TERM")  // Try SIGTERM first
            .arg("-f")
            .arg("llama-server")
            .output() {
                Ok(output) => {
                    info!("pkill -TERM llama-server executed, status: {:?}", output.status);
                }
                Err(e) => {
                    info!("Failed to execute pkill -TERM llama-server: {}", e);
                }
            }
            
        // Wait a bit more
        info!("Waiting after SIGTERM...");
        tokio::time::sleep(Duration::from_secs(1)).await;
        info!("SIGTERM wait completed");
            
        // Force kill with SIGKILL if still running
        info!("Force killing any remaining llama-server processes...");
        match Command::new("pkill")
            .arg("-KILL")  // SIGKILL
            .arg("-f")
            .arg("llama-server")
            .output() {
                Ok(output) => {
                    info!("pkill -KILL llama-server executed, status: {:?}", output.status);
                }
                Err(e) => {
                    info!("Failed to execute pkill -KILL llama-server: {}", e);
                }
            }
            
        // Also try to kill any llama.cpp processes
        info!("Killing any remaining llama.cpp processes...");
        match Command::new("pkill")
            .arg("-TERM")
            .arg("-f")
            .arg("llama.cpp")
            .output() {
                Ok(output) => {
                    info!("pkill -TERM llama.cpp executed, status: {:?}", output.status);
                }
                Err(e) => {
                    info!("Failed to execute pkill -TERM llama.cpp: {}", e);
                }
            }
            
        // Wait a bit more
        info!("Waiting after llama.cpp SIGTERM...");
        tokio::time::sleep(Duration::from_secs(1)).await;
        info!("llama.cpp SIGTERM wait completed");
            
        // Force kill with SIGKILL if still running
        info!("Force killing any remaining llama.cpp processes...");
        match Command::new("pkill")
            .arg("-KILL")
            .arg("-f")
            .arg("llama.cpp")
            .output() {
                Ok(output) => {
                    info!("pkill -KILL llama.cpp executed, status: {:?}", output.status);
                }
                Err(e) => {
                    info!("Failed to execute pkill -KILL llama.cpp: {}", e);
                }
            }
            
        info!("llama.cpp server stopped");
        Ok(())
    }

    pub async fn generate_embeddings(&self, texts: &[&str]) -> Result<Array2<f32>> {
        const BATCH_SIZE: usize = 10; // Process embeddings in batches
        let mut embeddings = Vec::new();
        
        // Track statistics
        let mut total_texts = 0;
        let mut successful_embeddings = 0;
        let mut failed_embeddings = 0;
        let mut size_stats = Vec::new();
        
        // Process texts in batches to reduce the number of HTTP requests
        for (batch_idx, batch) in texts.chunks(BATCH_SIZE).enumerate() {
            info!("Processing embedding batch {}/{} (size: {})", batch_idx + 1, (texts.len() + BATCH_SIZE - 1) / BATCH_SIZE, batch.len());
            
            for (i, text) in batch.iter().enumerate() {
                total_texts += 1;
                let original_length = text.len();
                
                // Add the document prefix for nomic embeddings
                let prompt = format!("search_document: {}", text);
                
                // Track size statistics
                size_stats.push(original_length);
                
                // Truncate very long texts to avoid server errors
                let truncated_prompt = if prompt.len() > 4000 {
                    &prompt[..4000]
                } else {
                    &prompt
                };
                
                let truncated_length = truncated_prompt.len();
                
                info!("Generating embedding {}/{} for text (original: {}, truncated: {})", 
                      batch_idx * BATCH_SIZE + i + 1, texts.len(), original_length, truncated_length);
                
                let start_time = std::time::Instant::now();
                let response = self.client.post(&format!("http://127.0.0.1:{}/embedding", self.server_port))
                    .json(&serde_json::json!({
                        "content": truncated_prompt
                    }))
                    .send()
                    .await;
                
                let duration = start_time.elapsed();
                
                match response {
                    Ok(response) => {
                        if response.status().is_success() {
                            info!("Successfully generated embedding {}/{} in {:?}", 
                                  batch_idx * BATCH_SIZE + i + 1, texts.len(), duration);
                            successful_embeddings += 1;
                            
                            // The response is an array containing one object with an embedding field
                            // The embedding field is an array containing one array of floats
                            match response.json::<Vec<serde_json::Value>>().await {
                                Ok(wrapper) => {
                                    if !wrapper.is_empty() {
                                        if let Some(embedding_array) = wrapper[0].get("embedding") {
                                            if let Some(inner_array) = embedding_array.as_array() {
                                                if !inner_array.is_empty() {
                                                    let embedding_values: Result<Vec<f32>, _> = inner_array[0]
                                                        .as_array()
                                                        .ok_or_else(|| anyhow::anyhow!("Invalid embedding format"))?
                                                        .iter()
                                                        .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| anyhow::anyhow!("Invalid float value")))
                                                        .collect();
                                                    match embedding_values {
                                                        Ok(values) => embeddings.push(values),
                                                        Err(e) => {
                                                            eprintln!("Failed to parse embedding values for text {}/{}: {}", 
                                                                     batch_idx * BATCH_SIZE + i + 1, texts.len(), e);
                                                            failed_embeddings += 1;
                                                            embeddings.push(vec![0.0; 768]);
                                                        }
                                                    }
                                                } else {
                                                    eprintln!("Empty embedding array for text {}/{}", 
                                                             batch_idx * BATCH_SIZE + i + 1, texts.len());
                                                    failed_embeddings += 1;
                                                    embeddings.push(vec![0.0; 768]);
                                                }
                                            } else {
                                                eprintln!("Invalid embedding format for text {}/{}", 
                                                         batch_idx * BATCH_SIZE + i + 1, texts.len());
                                                failed_embeddings += 1;
                                                embeddings.push(vec![0.0; 768]);
                                            }
                                        } else {
                                            eprintln!("No embedding field in response for text {}/{}", 
                                                     batch_idx * BATCH_SIZE + i + 1, texts.len());
                                            failed_embeddings += 1;
                                            embeddings.push(vec![0.0; 768]);
                                        }
                                    } else {
                                        eprintln!("Empty response for text {}/{}", 
                                                 batch_idx * BATCH_SIZE + i + 1, texts.len());
                                        failed_embeddings += 1;
                                        embeddings.push(vec![0.0; 768]);
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Failed to parse JSON response for text {}/{}: {}", 
                                             batch_idx * BATCH_SIZE + i + 1, texts.len(), e);
                                    failed_embeddings += 1;
                                    embeddings.push(vec![0.0; 768]);
                                }
                            }
                        } else {
                            // Log the error but continue with a zero vector
                            eprintln!("Failed to get embedding {}/{} for text (original: {}, truncated: {}) after {:?} - Status: {}", 
                                     batch_idx * BATCH_SIZE + i + 1, texts.len(), original_length, truncated_length, duration, response.status());
                            failed_embeddings += 1;
                            embeddings.push(vec![0.0; 768]);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to send request for embedding {}/{} for text (original: {}, truncated: {}) after {:?} - Error: {}", 
                                 batch_idx * BATCH_SIZE + i + 1, texts.len(), original_length, truncated_length, duration, e);
                        failed_embeddings += 1;
                        embeddings.push(vec![0.0; 768]);
                    }
                }
            }
        }
        
        // Report statistics
        if !size_stats.is_empty() {
            size_stats.sort();
            let min_size = size_stats[0];
            let max_size = size_stats[size_stats.len() - 1];
            let median_size = size_stats[size_stats.len() / 2];
            info!("Text size statistics - Min: {}, Max: {}, Median: {}", min_size, max_size, median_size);
        }
        
        info!("Embedding generation complete - Total: {}, Successful: {}, Failed: {}", 
              total_texts, successful_embeddings, failed_embeddings);
        
        // Convert to ndarray
        let num_embeddings = embeddings.len();
        if num_embeddings == 0 {
            return Err(anyhow::anyhow!("No embeddings generated"));
        }
        
        let dim = embeddings[0].len();
        let flat_embeddings: Vec<f32> = embeddings.into_iter().flatten().collect();
        
        Ok(Array::from_shape_vec((num_embeddings, dim), flat_embeddings)?)
    }
    
    pub async fn generate_query_embeddings(&self, texts: &[&str]) -> Result<Array2<f32>> {
        const BATCH_SIZE: usize = 10; // Process embeddings in batches
        let mut embeddings = Vec::new();
        
        // Track statistics
        let mut total_texts = 0;
        let mut successful_embeddings = 0;
        let mut failed_embeddings = 0;
        let mut size_stats = Vec::new();
        
        // Process texts in batches to reduce the number of HTTP requests
        for (batch_idx, batch) in texts.chunks(BATCH_SIZE).enumerate() {
            info!("Processing query embedding batch {}/{} (size: {})", batch_idx + 1, (texts.len() + BATCH_SIZE - 1) / BATCH_SIZE, batch.len());
            
            for (i, text) in batch.iter().enumerate() {
                total_texts += 1;
                let original_length = text.len();
                
                // Add the query prefix for nomic embeddings
                let prompt = format!("search_query: {}", text);
                
                // Track size statistics
                size_stats.push(original_length);
                
                // Truncate very long texts to avoid server errors
                let truncated_prompt = if prompt.len() > 4000 {
                    &prompt[..4000]
                } else {
                    &prompt
                };
                
                let truncated_length = truncated_prompt.len();
                
                info!("Generating query embedding {}/{} for text (original: {}, truncated: {})", 
                      batch_idx * BATCH_SIZE + i + 1, texts.len(), original_length, truncated_length);
                
                let start_time = std::time::Instant::now();
                let response = self.client.post(&format!("http://127.0.0.1:{}/embedding", self.server_port))
                    .json(&serde_json::json!({
                        "content": truncated_prompt
                    }))
                    .send()
                    .await;
                
                let duration = start_time.elapsed();
                
                match response {
                    Ok(response) => {
                        if response.status().is_success() {
                            info!("Successfully generated query embedding {}/{} in {:?}", 
                                  batch_idx * BATCH_SIZE + i + 1, texts.len(), duration);
                            successful_embeddings += 1;
                            
                            // The response is an array containing one object with an embedding field
                            // The embedding field is an array containing one array of floats
                            match response.json::<Vec<serde_json::Value>>().await {
                                Ok(wrapper) => {
                                    if !wrapper.is_empty() {
                                        if let Some(embedding_array) = wrapper[0].get("embedding") {
                                            if let Some(inner_array) = embedding_array.as_array() {
                                                if !inner_array.is_empty() {
                                                    let embedding_values: Result<Vec<f32>, _> = inner_array[0]
                                                        .as_array()
                                                        .ok_or_else(|| anyhow::anyhow!("Invalid embedding format"))?
                                                        .iter()
                                                        .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| anyhow::anyhow!("Invalid float value")))
                                                        .collect();
                                                    match embedding_values {
                                                        Ok(values) => embeddings.push(values),
                                                        Err(e) => {
                                                            eprintln!("Failed to parse embedding values for query text {}/{}: {}", 
                                                                     batch_idx * BATCH_SIZE + i + 1, texts.len(), e);
                                                            failed_embeddings += 1;
                                                            embeddings.push(vec![0.0; 768]);
                                                        }
                                                    }
                                                } else {
                                                    eprintln!("Empty embedding array for query text {}/{}", 
                                                             batch_idx * BATCH_SIZE + i + 1, texts.len());
                                                    failed_embeddings += 1;
                                                    embeddings.push(vec![0.0; 768]);
                                                }
                                            } else {
                                                eprintln!("Invalid embedding format for query text {}/{}", 
                                                         batch_idx * BATCH_SIZE + i + 1, texts.len());
                                                    failed_embeddings += 1;
                                                    embeddings.push(vec![0.0; 768]);
                                            }
                                        } else {
                                            eprintln!("No embedding field in response for query text {}/{}", 
                                                     batch_idx * BATCH_SIZE + i + 1, texts.len());
                                            failed_embeddings += 1;
                                            embeddings.push(vec![0.0; 768]);
                                        }
                                    } else {
                                        eprintln!("Empty response for query text {}/{}", 
                                                 batch_idx * BATCH_SIZE + i + 1, texts.len());
                                        failed_embeddings += 1;
                                        embeddings.push(vec![0.0; 768]);
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Failed to parse JSON response for query text {}/{}: {}", 
                                             batch_idx * BATCH_SIZE + i + 1, texts.len(), e);
                                    failed_embeddings += 1;
                                    embeddings.push(vec![0.0; 768]);
                                }
                            }
                        } else {
                            // Log the error but continue with a zero vector
                            eprintln!("Failed to get query embedding {}/{} for text (original: {}, truncated: {}) after {:?} - Status: {}", 
                                     batch_idx * BATCH_SIZE + i + 1, texts.len(), original_length, truncated_length, duration, response.status());
                            failed_embeddings += 1;
                            embeddings.push(vec![0.0; 768]);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to send request for query embedding {}/{} for text (original: {}, truncated: {}) after {:?} - Error: {}", 
                                 batch_idx * BATCH_SIZE + i + 1, texts.len(), original_length, truncated_length, duration, e);
                        failed_embeddings += 1;
                        embeddings.push(vec![0.0; 768]);
                    }
                }
            }
        }
        
        // Report statistics
        if !size_stats.is_empty() {
            size_stats.sort();
            let min_size = size_stats[0];
            let max_size = size_stats[size_stats.len() - 1];
            let median_size = size_stats[size_stats.len() / 2];
            info!("Query text size statistics - Min: {}, Max: {}, Median: {}", min_size, max_size, median_size);
        }
        
        info!("Query embedding generation complete - Total: {}, Successful: {}, Failed: {}", 
              total_texts, successful_embeddings, failed_embeddings);
        
        // Convert to ndarray
        let num_embeddings = embeddings.len();
        if num_embeddings == 0 {
            return Err(anyhow::anyhow!("No embeddings generated"));
        }
        
        let dim = embeddings[0].len();
        let flat_embeddings: Vec<f32> = embeddings.into_iter().flatten().collect();
        
        Ok(Array::from_shape_vec((num_embeddings, dim), flat_embeddings)?)
    }
}

pub struct DocumentMatcher {
    config: Config,
    xml_documents: HashMap<String, Vec<XmlElement>>,
    pdf_documents: HashMap<String, Vec<PdfBlock>>,
    embedder: EmbeddingGenerator,
    qdrant_client: Qdrant,
    xml_collection_name: String,
    xml_element_map: HashMap<u64, (String, String)>, // map point id to (doc_id, element_id)
    qdrant_container_id: Option<String>,
}

impl DocumentMatcher {
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing DocumentMatcher");
        
        // Start Qdrant Docker container
        let container_id = Self::start_qdrant_container().await?;
        
        // Wait for Qdrant to be ready
        Self::wait_for_qdrant_ready().await?;
        
        let mut embedder = EmbeddingGenerator::new(config.model_path.to_string_lossy().to_string())?;
        
        // Start the embedding server
        embedder.start_server().await?;

        // Initialize Qdrant client
        let qdrant_client = Qdrant::from_url("http://localhost:6336").build()?;
        
        let mut matcher = DocumentMatcher {
            config,
            xml_documents: HashMap::new(),
            pdf_documents: HashMap::new(),
            embedder,
            qdrant_client,
            xml_collection_name: "xml_documents".to_string(),
            xml_element_map: HashMap::new(),
            qdrant_container_id: Some(container_id),
        };

        // Create collection if it doesn't exist
        matcher.create_collection().await?;
        
        matcher.load_xml_documents().await?;
        matcher.build_xml_index().await?;

        Ok(matcher)
    }

    async fn start_qdrant_container() -> Result<String> {
        info!("Starting Qdrant Docker container...");
        
        // Check if a container with the same name is already running and stop it
        Self::stop_existing_qdrant_container().await?;
        
        // Pull the latest Qdrant image
        let pull_output = tokio::task::spawn_blocking(|| {
            Command::new("docker")
                .arg("pull")
                .arg("qdrant/qdrant")
                .output()
        }).await??;
            
        if !pull_output.status.success() {
            return Err(anyhow::anyhow!("Failed to pull Qdrant Docker image: {}", String::from_utf8_lossy(&pull_output.stderr)));
        }
        
        // Start the Qdrant container
        let start_output = tokio::task::spawn_blocking(|| {
            Command::new("docker")
                .arg("run")
                .arg("-d")
                .arg("--name")
                .arg("qdrant-matching-app")
                .arg("-p")
                .arg("6335:6333")
                .arg("-p")
                .arg("6336:6334")
                .arg("-v")
                .arg(format!("{}:/qdrant/storage:z", std::env::current_dir()?.join("qdrant_storage").to_string_lossy()))
                .arg("qdrant/qdrant")
                .output()
        }).await??;
            
        if !start_output.status.success() {
            return Err(anyhow::anyhow!("Failed to start Qdrant Docker container: {}", String::from_utf8_lossy(&start_output.stderr)));
        }
        
        let container_id = String::from_utf8(start_output.stdout)?.trim().to_string();
        info!("Qdrant Docker container started with ID: {}", container_id);
        
        Ok(container_id)
    }
    
    async fn stop_existing_qdrant_container() -> Result<()> {
        // Check if container exists
        let inspect_output = tokio::task::spawn_blocking(|| {
            Command::new("docker")
                .arg("inspect")
                .arg("qdrant-matching-app")
                .output()
        }).await??;
        
        // If container exists, stop and remove it
        if inspect_output.status.success() {
            info!("Stopping existing Qdrant Docker container...");
            
            // Stop the container
            let stop_output = tokio::task::spawn_blocking(|| {
                Command::new("docker")
                    .arg("stop")
                    .arg("qdrant-matching-app")
                    .output()
            }).await??;
            
            if !stop_output.status.success() {
                info!("Failed to stop existing Qdrant Docker container: {}", String::from_utf8_lossy(&stop_output.stderr));
            } else {
                info!("Existing Qdrant Docker container stopped");
            }
            
            // Remove the container
            let remove_output = tokio::task::spawn_blocking(|| {
                Command::new("docker")
                    .arg("rm")
                    .arg("qdrant-matching-app")
                    .output()
            }).await??;
            
            if !remove_output.status.success() {
                info!("Failed to remove existing Qdrant Docker container: {}", String::from_utf8_lossy(&remove_output.stderr));
            } else {
                info!("Existing Qdrant Docker container removed");
            }
        }
        
        Ok(())
    }
    
    async fn wait_for_qdrant_ready() -> Result<()> {
        info!("Waiting for Qdrant to be ready...");
        let client = reqwest::Client::new();
        let max_retries = 30;
        let mut retries = 0;
        
        loop {
            match client.get("http://localhost:6335/readyz").send().await {
                Ok(response) if response.status().is_success() => {
                    info!("Qdrant is ready");
                    return Ok(());
                }
                _ => {
                    if retries >= max_retries {
                        return Err(anyhow::anyhow!("Qdrant failed to start after {} retries", max_retries));
                    }
                    retries += 1;
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    async fn create_collection(&mut self) -> Result<()> {
        // Check if collection exists
        let collections = self.qdrant_client.list_collections().await?;
        if !collections.collections.iter().any(|c| c.name == self.xml_collection_name) {
            // Create collection with 256 dimensions for Matryoshka embeddings
            self.qdrant_client
                .create_collection(
                    CreateCollectionBuilder::new(&self.xml_collection_name)
                        .vectors_config(VectorParamsBuilder::new(256, Distance::Dot)),
                )
                .await?;
            info!("Created Qdrant collection: {}", self.xml_collection_name);
        } else {
            info!("Qdrant collection already exists: {}", self.xml_collection_name);
        }
        
        Ok(())
    }

    fn get_embedding_cache_path(&self) -> std::path::PathBuf {
        // Use project root directory for cache to avoid accidental deletion with outputs
        std::env::current_dir()
            .unwrap_or_else(|_| self.config.output_dir.clone())
            .join("xml_embeddings_cache.json")
    }

    async fn save_embedding_cache(&self, embeddings: &Array2<f32>, element_map: &HashMap<u64, (String, String)>) -> Result<()> {
        let cache_path = self.get_embedding_cache_path();
        info!("Saving embedding cache to {:?}", cache_path);
        
        // Truncate embeddings to 256 dimensions for consistency
        let truncated_embeddings = if embeddings.shape()[1] > 256 {
            embeddings.clone().slice_move(ndarray::s![.., ..256])
        } else {
            embeddings.clone()
        };
        
        // Create a serializable representation of the embeddings
        // Ensure the array is contiguous before getting a slice
        let contiguous_embeddings = truncated_embeddings.to_owned();
        let cache_data = serde_json::json!({
            "embeddings": contiguous_embeddings.as_slice().unwrap(),
            "shape": [contiguous_embeddings.shape()[0], contiguous_embeddings.shape()[1]],
            "element_map": element_map,
            "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or(std::time::Duration::from_secs(0)).as_secs()
        });
        
        fs::write(&cache_path, serde_json::to_string_pretty(&cache_data)?)?;
        info!("Saved embedding cache with {} embeddings", truncated_embeddings.shape()[0]);
        Ok(())
    }

    async fn load_embedding_cache(&self) -> Result<Option<(Array2<f32>, HashMap<u64, (String, String)>)>> {
        let cache_path = self.get_embedding_cache_path();
        if !cache_path.exists() {
            return Ok(None);
        }
        
        info!("Loading embedding cache from {:?}", cache_path);
        let content = fs::read_to_string(&cache_path)?;
        let cache_data: serde_json::Value = serde_json::from_str(&content)?;
        
        // Extract embeddings
        if let Some(embeddings_array) = cache_data.get("embeddings").and_then(|v| v.as_array()) {
            if let Some(shape_array) = cache_data.get("shape").and_then(|v| v.as_array()) {
                if shape_array.len() == 2 {
                    let rows = shape_array[0].as_u64().unwrap_or(0) as usize;
                    let cols = shape_array[1].as_u64().unwrap_or(0) as usize;
                    
                    // Convert JSON array to Vec<f32>
                    let mut embeddings_flat = Vec::new();
                    for value in embeddings_array {
                        if let Some(f) = value.as_f64() {
                            embeddings_flat.push(f as f32);
                        } else {
                            // If we can't parse a value, the cache is invalid
                            let _ = fs::remove_file(&cache_path);
                            return Ok(None);
                        }
                    }
                    
                    if let Ok(embeddings) = Array::from_shape_vec((rows, cols), embeddings_flat) {
                        // Extract element map
                        if let Some(element_map_json) = cache_data.get("element_map") {
                            if let Ok(element_map) = serde_json::from_value(element_map_json.clone()) {
                                info!("Loaded embedding cache with {} embeddings", rows);
                                return Ok(Some((embeddings, element_map)));
                            }
                        }
                    }
                }
            }
        }
        
        // If we couldn't load the cache properly, remove it
        let _ = fs::remove_file(&cache_path);
        Ok(None)
    }

    async fn build_xml_index(&mut self) -> Result<()> {
        info!("Building Qdrant index for XML documents...");
        
        // First, try to load from cache
        if let Some((cached_embeddings, cached_element_map)) = self.load_embedding_cache().await? {
            info!("Using cached embeddings for XML index");
            self.xml_element_map = cached_element_map;
            
            // Create points for Qdrant from cached embeddings
            let mut points = Vec::new();
            let mut all_xml_texts = Vec::new();
            
            // Rebuild text list to match cached embeddings
            for (_doc_id, elements) in &self.xml_documents {
                for element in elements {
                    all_xml_texts.push(element.text.as_str());
                }
            }
            
            // Truncate cached embeddings to 256 dimensions as specified in Matching.md
            let truncated_cached_embeddings = if cached_embeddings.shape()[1] > 256 {
                cached_embeddings.slice_move(ndarray::s![.., ..256])
            } else {
                cached_embeddings
            };
            
            let point_count = std::cmp::min(truncated_cached_embeddings.shape()[0], all_xml_texts.len());
            for (i, row) in truncated_cached_embeddings.slice(ndarray::s![0..point_count, ..]).rows().into_iter().enumerate() {
                if i < self.xml_element_map.len() {
                    let vector: Vec<f32> = row.to_vec();
                    
                    // Find the corresponding element
                    if let Some((doc_id, element_id)) = self.xml_element_map.get(&(i as u64 + 1)) {
                        let payload_json = serde_json::json!({
                            "doc_id": doc_id,
                            "element_id": element_id,
                            "text": if i < all_xml_texts.len() { all_xml_texts[i] } else { "" }
                        });
                        
                        let payload: qdrant_client::Payload = match payload_json.try_into() {
                            Ok(p) => p,
                            Err(e) => {
                                eprintln!("Failed to convert payload to Qdrant payload: {:?}", e);
                                // Create a minimal payload as fallback
                                serde_json::json!({
                                    "doc_id": doc_id,
                                    "element_id": element_id,
                                    "text": ""
                                }).try_into().unwrap_or_else(|_| serde_json::json!({
                                    "doc_id": "",
                                    "element_id": "",
                                    "text": ""
                                }).try_into().expect("Failed to create fallback payload"))
                            }
                        };
                        
                        let point = PointStruct::new(
                            i as u64 + 1,
                            vector,
                            payload,
                        );
                        
                        points.push(point);
                    }
                }
            }
            
            info!("Upserting {} cached points to Qdrant", points.len());
            let start_time = std::time::Instant::now();
            
            // Upsert points to Qdrant
            self.qdrant_client
                .upsert_points(UpsertPointsBuilder::new(&self.xml_collection_name, points.clone()).wait(true))
                .await?;
            
            let duration = start_time.elapsed();
            info!("Upserted {} cached points to Qdrant in {:?}", points.len(), duration);
            
            info!("Qdrant index built with {} XML elements from cache.", points.len());
            return Ok(());
        }
        
        // If no cache, build from scratch
        let mut all_xml_texts = Vec::new();
        let mut element_map = HashMap::new();
        let mut current_point_id = 1u64;

        for (doc_id, elements) in &self.xml_documents {
            for element in elements {
                all_xml_texts.push(element.text.as_str());
                element_map.insert(current_point_id, (doc_id.clone(), element.id.clone()));
                current_point_id += 1;
            }
        }

        if all_xml_texts.is_empty() {
            return Err(MatchingError::NoXmlDocuments.into());
        }

        info!("Generating embeddings for {} XML elements", all_xml_texts.len());
        let start_time = std::time::Instant::now();
        
        // Generate embeddings with document prefix
        let embeddings = self.embedder.generate_embeddings(&all_xml_texts).await?;
        
        let duration = start_time.elapsed();
        info!("Generated embeddings for {} XML elements in {:?}", all_xml_texts.len(), duration);
        
        // Save embeddings to cache
        self.save_embedding_cache(&embeddings, &element_map).await?;
        
        // Truncate to 256 dimensions as specified in Matching.md
        let truncated_embeddings = embeddings.slice_move(ndarray::s![.., ..256]);
        
        // Create points for Qdrant
        let mut points = Vec::new();
        let mut point_id = 1u64;
        
        for (i, row) in truncated_embeddings.rows().into_iter().enumerate() {
            let vector: Vec<f32> = row.to_vec();
            let (doc_id, element_id) = &element_map[&(i as u64 + 1)];
            
            let payload: qdrant_client::Payload = serde_json::json!({
                "doc_id": doc_id,
                "element_id": element_id,
                "text": all_xml_texts[i]
            }).try_into().unwrap_or_else(|_| serde_json::json!({
                "doc_id": doc_id,
                "element_id": element_id,
                "text": ""
            }).try_into().expect("Failed to create payload"));
            
            let point = PointStruct::new(
                point_id,
                vector,
                payload,
            );
            
            points.push(point);
            point_id += 1;
        }
        
        info!("Upserting {} points to Qdrant", points.len());
        let start_time = std::time::Instant::now();
        
        // Upsert points to Qdrant
        self.qdrant_client
            .upsert_points(UpsertPointsBuilder::new(&self.xml_collection_name, points.clone()).wait(true))
            .await?;
        
        let duration = start_time.elapsed();
        info!("Upserted {} points to Qdrant in {:?}", points.len(), duration);
        
        self.xml_element_map = element_map;

        info!("Qdrant index built with {} XML elements.", points.len());
        Ok(())
    }
    
    pub async fn process_all_pdfs(&mut self) -> Result<()> {
        info!("Loading PDF documents");
        self.load_pdf_documents().await?;
        
        info!("Processing PDF documents for matching");
        self.match_documents().await?;
        
        info!("Stopping embedding server");
        // Stop the embedding server
        self.embedder.stop_server().await?;
        
        info!("PDF processing completed successfully");
        Ok(())
    }
    
    pub async fn stop_qdrant_container(&mut self) -> Result<()> {
        info!("Stopping Qdrant Docker container...");
        
        // Stop the container by name
        info!("Sending stop command to Qdrant container...");
        let stop_result = tokio::time::timeout(
            Duration::from_secs(30),
            tokio::task::spawn_blocking(|| {
                Command::new("docker")
                    .arg("stop")
                    .arg("qdrant-matching-app")
                    .output()
            })
        ).await;
        
        match stop_result {
            Ok(Ok(Ok(output))) => {
                if !output.status.success() {
                    // If the container is not running, that's fine - we'll try to remove it anyway
                    info!("Qdrant Docker container was not running or failed to stop: {}", String::from_utf8_lossy(&output.stderr));
                } else {
                    info!("Qdrant Docker container stopped");
                }
            }
            Ok(Ok(Err(e))) => {
                info!("Failed to execute stop command: {}", e);
            }
            Ok(Err(e)) => {
                info!("Failed to spawn stop command: {}", e);
            }
            Err(_) => {
                info!("Stop command timed out");
            }
        }
        info!("Stop command completed");
        
        // Remove the container by name
        info!("Removing Qdrant container...");
        let remove_result = tokio::time::timeout(
            Duration::from_secs(30),
            tokio::task::spawn_blocking(|| {
                Command::new("docker")
                    .arg("rm")
                    .arg("qdrant-matching-app")
                    .output()
            })
        ).await;
        
        match remove_result {
            Ok(Ok(Ok(output))) => {
                if !output.status.success() {
                    // If the container doesn't exist, that's fine
                    info!("Qdrant Docker container was not found or failed to remove: {}", String::from_utf8_lossy(&output.stderr));
                } else {
                    info!("Qdrant Docker container removed");
                }
            }
            Ok(Ok(Err(e))) => {
                info!("Failed to execute remove command: {}", e);
            }
            Ok(Err(e)) => {
                info!("Failed to spawn remove command: {}", e);
            }
            Err(_) => {
                info!("Remove command timed out");
            }
        }
        info!("Remove command completed");
        
        self.qdrant_container_id = None;
        info!("Qdrant container shutdown completed");
        Ok(())
    }
    
    async fn load_xml_documents(&mut self) -> Result<()> {
        let xml_dir = self.config.xml_documents_dir.clone();
        if !xml_dir.exists() {
            return Err(MatchingError::InvalidConfig(format!("XML documents directory does not exist: {:?}", xml_dir)).into());
        }
        
        self.load_xml_documents_recursive(&xml_dir)?;
        
        info!("Loaded {} XML documents", self.xml_documents.len());
        Ok(())
    }
    
    fn load_xml_documents_recursive(&mut self, dir: &Path) -> Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    self.load_xml_documents_recursive(&path)?;
                } else if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    info!("Loading XML document: {:?}", path);
                    let file_content = fs::read_to_string(&path)?;
                    let elements: Vec<XmlElement> = serde_json::from_str(&file_content)?;
                    
                    let filename = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown_xml")
                        .to_string();
                    
                    self.xml_documents.insert(filename, elements);
                }
            }
        }
        Ok(())
    }
    
    async fn load_pdf_documents(&mut self) -> Result<()> {
        let pdf_dir = self.config.pdf_input_dir.clone();
        if !pdf_dir.exists() {
            return Err(MatchingError::InvalidConfig(format!("PDF input directory does not exist: {:?}", pdf_dir)).into());
        }
        
        self.load_pdf_documents_recursive(&pdf_dir)?;
        
        info!("Loaded {} PDF documents", self.pdf_documents.len());
        Ok(())
    }
    
    fn load_pdf_documents_recursive(&mut self, dir: &Path) -> Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    self.load_pdf_documents_recursive(&path)?;
                } else if path.extension().and_then(|s| s.to_str()) == Some("json") 
                    && path.to_string_lossy().ends_with("_processed.json") {
                    info!("Loading PDF document: {:?}", path);
                    let file_content = fs::read_to_string(&path)?;
                    let blocks: Vec<PdfBlock> = serde_json::from_str(&file_content)?;
                    
                    let filename = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown_pdf")
                        .to_string();
                    
                    self.pdf_documents.insert(filename, blocks);
                }
            }
        }
        Ok(())
    }
    
    async fn match_documents(&mut self) -> Result<()> {
        info!("Starting document matching process");
        
        if self.xml_documents.is_empty() {
            return Err(MatchingError::NoXmlDocuments.into());
        }
        
        if self.pdf_documents.is_empty() {
            return Err(MatchingError::NoPdfDocuments.into());
        }
        
        // Collect PDF document names and blocks to avoid borrowing issues
        let pdf_data: Vec<(String, Vec<PdfBlock>)> = self.pdf_documents
            .iter()
            .map(|(name, blocks)| (name.clone(), blocks.clone()))
            .collect();
        
        info!("Processing {} PDF documents", pdf_data.len());
        
        for (pdf_name, pdf_blocks) in pdf_data {
            info!("Matching PDF document: {}", pdf_name);
            let start_time = std::time::Instant::now();
            
            let document_similarities = self.calculate_document_similarities(&pdf_blocks).await?;
            
            if document_similarities.is_empty() {
                return Err(MatchingError::InvalidConfig("No document similarities found".to_string()).into());
            }
            
            let best_match = document_similarities
                .iter()
                .max_by(|a, b| a.similarity.partial_cmp(&b.similarity).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or(MatchingError::InvalidConfig("No best match found".to_string()))?;
            
            info!("Best matching XML document for {}: {} (score: {})", 
                  pdf_name, best_match.xml_document_id, best_match.similarity);
            
            let best_xml_elements = self.xml_documents.get(&best_match.xml_document_id)
                .ok_or_else(|| MatchingError::InvalidConfig(format!("XML document not found: {}", best_match.xml_document_id)))?
                .clone();
            let (matches, statistics, top_phrase_matches) = self.match_paragraphs(&best_xml_elements, &pdf_blocks, &best_match.xml_document_id).await?;
            
            let document_match = DocumentMatch {
                pdf_document_id: pdf_name.clone(),
                best_match: (*best_match).clone(),
                all_document_scores: document_similarities,
                paragraph_matches: matches,
                top_phrase_matches,
                statistics,
            };
            
            self.save_results(&pdf_name, &document_match).await?;
            
            let duration = start_time.elapsed();
            info!("Completed matching for {} in {:?}", pdf_name, duration);
        }
        
        info!("Document matching process completed successfully");
        Ok(())
    }
    
    async fn calculate_document_similarities(&self, pdf_blocks: &[PdfBlock]) -> Result<Vec<DocumentMatchInfo>> {
        info!("Calculating document similarities with enhanced approach");
        let start_time = std::time::Instant::now();
        
        // Take first 10 blocks as specified in Matching.md
        let first_10_pdf_texts: Vec<&str> = pdf_blocks.iter().take(10).map(|b| b.text.as_str()).collect();
        
        // Precompute text statistics for PDF blocks
        let pdf_stats: Vec<TextStatistics> = first_10_pdf_texts.iter().map(|text| TextStatistics::new(text)).collect();
        
        info!("Generating query embeddings for first 10 PDF blocks");
        
        // Generate query embeddings with search_query prefix for the first 10 PDF blocks
        let pdf_embeddings = self.embedder.generate_query_embeddings(&first_10_pdf_texts).await?;
        
        // Truncate to 256 dimensions
        let truncated_pdf_embeddings = pdf_embeddings.slice_move(ndarray::s![.., ..256]);
        
        let mut similarities = Vec::new();
        
        // For each XML document, compare the first 10 PDF embeddings against the first 10 XML embeddings
        for (doc_id, xml_elements) in &self.xml_documents {
            // Take first 10 elements from the XML document
            let first_10_xml_texts: Vec<&str> = xml_elements.iter().take(10).map(|e| e.text.as_str()).collect();
            
            // Precompute text statistics for XML elements
            let xml_stats: Vec<TextStatistics> = first_10_xml_texts.iter().map(|text| TextStatistics::new(text)).collect();
            
            if first_10_xml_texts.is_empty() {
                similarities.push(DocumentMatchInfo {
                    xml_document_id: doc_id.clone(),
                    similarity: 0.0,
                    top_phrase_pairs: vec![],
                });
                continue;
            }
            
            info!("Generating document embeddings for first 10 elements of XML document: {}", doc_id);
            
            // Generate document embeddings with search_document prefix for the first 10 XML elements
            let xml_embeddings = self.embedder.generate_embeddings(&first_10_xml_texts).await?;
            
            // Truncate to 256 dimensions
            let truncated_xml_embeddings = xml_embeddings.slice_move(ndarray::s![.., ..256]);
            
            // Calculate max similarity for each PDF embedding (max pooling approach)
            let mut max_composite_scores = Vec::new();
            let mut top_phrase_pairs = Vec::new();
            
            // For each PDF embedding, find the maximum composite score with any XML embedding
            for (pdf_idx, pdf_row) in truncated_pdf_embeddings.rows().into_iter().enumerate() {
                let pdf_vector: Vec<f32> = pdf_row.to_vec();
                let pdf_text = first_10_pdf_texts[pdf_idx];
                let pdf_stat = &pdf_stats[pdf_idx];
                
                // Skip zero vectors
                if pdf_vector.iter().all(|&x| x == 0.0) {
                    max_composite_scores.push(0.0);
                    continue;
                }
                
                let mut max_composite_score = f32::NEG_INFINITY;
                let mut best_xml_idx = 0;
                let mut best_semantic_similarity = 0.0;
                let mut best_length_similarity = 0.0;
                let mut best_word_similarity = 0.0;
                
                // Find the maximum composite score with any XML embedding
                for (xml_idx, xml_row) in truncated_xml_embeddings.rows().into_iter().enumerate() {
                    let xml_vector: Vec<f32> = xml_row.to_vec();
                    let _xml_text = first_10_xml_texts[xml_idx];
                    let xml_stat = &xml_stats[xml_idx];
                    
                    // Skip zero vectors
                    if xml_vector.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    
                    // Calculate semantic similarity (cosine similarity)
                    let semantic_similarity = cosine_similarity(&pdf_vector, &xml_vector);
                    
                    // Calculate structural similarities
                    let length_similarity = pdf_stat.length_similarity(xml_stat);
                    let word_similarity = pdf_stat.word_similarity(xml_stat);
                    
                    // Calculate composite score
                    let composite_score = calculate_composite_score(semantic_similarity, length_similarity, word_similarity);
                    
                    if composite_score > max_composite_score {
                        max_composite_score = composite_score;
                        best_xml_idx = xml_idx;
                        best_semantic_similarity = semantic_similarity;
                        best_length_similarity = length_similarity;
                        best_word_similarity = word_similarity;
                    }
                }
                
                // If no valid similarity was found, use 0.0
                if max_composite_score == f32::NEG_INFINITY {
                    max_composite_score = 0.0;
                } else if pdf_idx < pdf_blocks.len() && best_xml_idx < xml_elements.len() {
                    // Store the top phrase pair with all similarity metrics
                    top_phrase_pairs.push(TopPhrasePair {
                        xml_element_id: xml_elements[best_xml_idx].id.clone(),
                        xml_text: first_10_xml_texts[best_xml_idx].to_string(),
                        pdf_block_id: pdf_blocks[pdf_idx].id.clone(),
                        pdf_text: pdf_text.to_string(),
                        similarity: max_composite_score, // Using composite score as the similarity
                    });
                    
                    // Log the individual components for debugging
                    info!("PDF block {}: semantic={}, length={}, word={}, composite={}", 
                          pdf_idx, best_semantic_similarity, best_length_similarity, best_word_similarity, max_composite_score);
                }
                
                max_composite_scores.push(max_composite_score);
            }
            
            // Calculate average of maximum composite scores (max pooling approach)
            let avg_max_composite_score = if !max_composite_scores.is_empty() {
                max_composite_scores.iter().sum::<f32>() / max_composite_scores.len() as f32
            } else {
                0.0
            };
            
            // Calculate filename similarity
            let filename_sim = filename_similarity(&doc_id, &self.pdf_documents.keys().next().unwrap_or(&String::new()));
            
            // Combine composite score with filename similarity (weighted average)
            let final_score = 0.9 * avg_max_composite_score + 0.1 * filename_sim;
            
            info!("Document {}: composite score={}, filename similarity={}, final score={}", 
                  doc_id, avg_max_composite_score, filename_sim, final_score);
            
            similarities.push(DocumentMatchInfo {
                xml_document_id: doc_id.clone(),
                similarity: final_score,
                top_phrase_pairs,
            });
        }
        
        similarities.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        
        info!("Found {} document similarities", similarities.len());
        if !similarities.is_empty() {
            info!("Best match: {} (score: {})", similarities[0].xml_document_id, similarities[0].similarity);
        }
        
        let duration = start_time.elapsed();
        info!("Calculated document similarities in {:?}", duration);
        
        Ok(similarities)
    }

    async fn match_paragraphs(&mut self, xml_elements: &[XmlElement], pdf_blocks: &[PdfBlock], xml_document_id: &str) -> Result<(Vec<ElementMatchResult>, MatchStatistics, Vec<TopPhraseMatch>)> {
        info!("Matching {} XML elements with {} PDF blocks using enhanced approach", xml_elements.len(), pdf_blocks.len());
        let start_time = std::time::Instant::now();
        
        let pdf_texts: Vec<&str> = pdf_blocks.iter().map(|b| b.text.as_str()).collect();
        
        // Precompute text statistics for PDF blocks
        let pdf_stats: Vec<TextStatistics> = pdf_texts.iter().map(|text| TextStatistics::new(text)).collect();
        
        // Generate query embeddings with search_query prefix
        let pdf_embeddings = self.embedder.generate_query_embeddings(&pdf_texts).await?;
        
        // Check if embeddings are all zeros (which indicates a problem)
        let is_all_zeros = pdf_embeddings.iter().all(|&x| x == 0.0);
        if is_all_zeros {
            info!("Warning: All PDF embeddings are zero, this indicates an embedding generation problem");
        }
        
        // Truncate to 256 dimensions
        let truncated_pdf_embeddings = pdf_embeddings.slice_move(ndarray::s![.., ..256]);

        let k = self.config.top_k_matches;
        let mut xml_to_pdf_matches: HashMap<String, Vec<(f32, &PdfBlock)>> = HashMap::new();

        // Track embedding generation statistics
        let mut successful_embeddings = 0;
        let mut failed_embeddings = 0;

        // For each PDF block, search for similar XML elements
        for (pdf_idx, row) in truncated_pdf_embeddings.rows().into_iter().enumerate() {
            let query_vector: Vec<f32> = row.to_vec();
            
            // Check if this query vector is all zeros
            let is_query_zero = query_vector.iter().all(|&x| x == 0.0);
            if is_query_zero {
                info!("Skipping zero query vector for PDF block {} (text length: {})", pdf_idx, pdf_texts[pdf_idx].len());
                failed_embeddings += 1;
                continue;
            } else {
                successful_embeddings += 1;
            }
            
            let pdf_block = &pdf_blocks[pdf_idx];
            let pdf_stat = &pdf_stats[pdf_idx];
            
            // Search in Qdrant
            let search_result = self.qdrant_client
                .query(
                    QueryPointsBuilder::new(&self.xml_collection_name)
                        .query(query_vector)
                        .limit(k as u64 * 2) // Get more results to allow for filtering
                        .with_payload(true)
                )
                .await?;
            
            // For each result, we'll enhance the similarity score with structural features
            for result in search_result.result {
                let semantic_similarity = result.score;
                
                // Get the element_id and text from payload
                let element_id = if let Some(payload) = result.payload.get("element_id") {
                    if let Some(str_val) = payload.as_str() {
                        str_val.to_string()
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };
                
                let xml_text = if let Some(payload) = result.payload.get("text") {
                    if let Some(str_val) = payload.as_str() {
                        str_val.to_string()
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };
                
                // Calculate structural similarities
                let xml_stat = TextStatistics::new(&xml_text);
                let length_similarity = pdf_stat.length_similarity(&xml_stat);
                let word_similarity = pdf_stat.word_similarity(&xml_stat);
                
                // Calculate composite score
                let composite_score = calculate_composite_score(semantic_similarity, length_similarity, word_similarity);
                
                xml_to_pdf_matches.entry(element_id)
                    .or_default()
                    .push((composite_score, pdf_block));
            }
        }

        let mut results = Vec::new();
        let mut matched_elements_count = 0;
        let mut matches_above_01 = 0;
        let mut matches_above_03 = 0;

        for xml_element in xml_elements {
            let top_matches = if let Some(matches) = xml_to_pdf_matches.get_mut(&xml_element.id) {
                matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                matches.truncate(k);
                // Always count as matched since we're returning top K results regardless of threshold
                matched_elements_count += 1;
                
                // Count matches above thresholds
                for (similarity, _) in matches.iter() {
                    if *similarity >= 0.1 {
                        matches_above_01 += 1;
                    }
                    if *similarity >= 0.3 {
                        matches_above_03 += 1;
                    }
                }
                
                matches.iter().map(|(sim, block)| MatchResult {
                    pdf_block_id: block.id.clone(),
                    similarity: *sim,
                    pdf_text: block.text.clone(),
                }).collect()
            } else {
                Vec::new()
            };

            results.push(ElementMatchResult {
                xml_element_id: xml_element.id.clone(),
                xml_text: xml_element.text.clone(),
                top_matches,
            });
        }

        // Collect top phrase matches across all elements
        let mut all_matches = Vec::new();
        for xml_element in xml_elements {
            if let Some(matches) = xml_to_pdf_matches.get(&xml_element.id) {
                // Clone the matches to avoid borrow issues
                let matches_clone = matches.clone();
                for (similarity, pdf_block) in matches_clone {
                    all_matches.push(TopPhraseMatch {
                        xml_document_id: xml_document_id.to_string(),
                        xml_element_id: xml_element.id.clone(),
                        xml_text: xml_element.text.clone(),
                        pdf_block_id: pdf_block.id.clone(),
                        pdf_text: pdf_block.text.clone(),
                        similarity,
                    });
                }
            }
        }
        
        // Sort by similarity and take top 10
        all_matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        let top_phrase_matches = all_matches.into_iter().take(10).collect();

        let statistics = MatchStatistics {
            total_xml_documents: self.xml_documents.len(),
            total_xml_elements: xml_elements.len(),
            total_pdf_blocks: pdf_blocks.len(),
            matched_elements: matched_elements_count,
            unmatched_elements: xml_elements.len() - matched_elements_count,
            match_threshold: self.config.similarity_threshold,
            top_k_matches: self.config.top_k_matches,
            matches_above_01,
            matches_above_03,
            top_phrase_matches: vec![], // We'll handle this separately in the DocumentMatch struct
        };
        
        let duration = start_time.elapsed();
        info!("Completed matching in {:?} - Matched {} out of {} elements", 
              duration, matched_elements_count, xml_elements.len());
        info!("Embedding generation: {} successful, {} failed", successful_embeddings, failed_embeddings);
        info!("Matches above 0.1: {}, Matches above 0.3: {}", matches_above_01, matches_above_03);

        Ok((results, statistics, top_phrase_matches))
    }
    
    /*
    /// Calculate filename similarities between all PDF and XML documents
    fn calculate_filename_similarities(&self) -> HashMap<String, HashMap<String, f32>> {
        info!("Calculating filename similarities");
        
        let mut filename_similarities = HashMap::new();
        
        // For each PDF document, calculate similarity with all XML documents
        for (pdf_id, _) in &self.pdf_documents {
            let mut doc_similarities = HashMap::new();
            
            for (xml_id, _) in &self.xml_documents {
                let similarity = filename_similarity(pdf_id, xml_id);
                doc_similarities.insert(xml_id.clone(), similarity);
            }
            
            filename_similarities.insert(pdf_id.clone(), doc_similarities);
        }
        
        info!("Calculated filename similarities for {} PDF documents", filename_similarities.len());
        filename_similarities
    }
    */
    
    async fn save_results(&self, pdf_name: &str, document_match: &DocumentMatch) -> Result<()> {
        let output_path = self.config.output_dir.join(format!("{}_matches.json", pdf_name));
        let json_content = serde_json::to_string_pretty(document_match)?;
        fs::write(&output_path, json_content)?;
        info!("Saved matching results to {:?}", output_path);
        
        info!("Matching statistics for {}:", pdf_name);
        info!("  Total XML documents processed: {}", document_match.statistics.total_xml_documents);
        info!("  Total XML elements processed: {}", document_match.statistics.total_xml_elements);
        info!("  Total PDF blocks processed: {}", document_match.statistics.total_pdf_blocks);
        info!("  Matched elements: {}", document_match.statistics.matched_elements);
        info!("  Unmatched elements: {}", document_match.statistics.unmatched_elements);
        info!("  Match threshold: {}", document_match.statistics.match_threshold);
        info!("  Top K matches: {}", document_match.statistics.top_k_matches);
        info!("  Matches above 0.1: {}", document_match.statistics.matches_above_01);
        info!("  Matches above 0.3: {}", document_match.statistics.matches_above_03);
        info!("  Best matching XML document: {} (score: {})", 
              document_match.best_match.xml_document_id, document_match.best_match.similarity);
        
        info!("  All document similarities:");
        for (i, doc_sim) in document_match.all_document_scores.iter().enumerate() {
            info!("    {}. {} (score: {})", i + 1, doc_sim.xml_document_id, doc_sim.similarity);
        }
        
        Ok(())
    }
}

impl Drop for DocumentMatcher {
    fn drop(&mut self) {
        info!("DocumentMatcher is being dropped");
        // Any cleanup code can go here if needed
        // For example, we could try to explicitly close the Qdrant client connection
        // if there was a method to do so
        
        // In a real implementation, we would stop the Qdrant container here
        // but since this is in a Drop handler, we can't use async code
        // The stop_qdrant_container method should be called explicitly before dropping
        info!("DocumentMatcher drop completed");
    }
}

impl Drop for EmbeddingGenerator {
    fn drop(&mut self) {
        info!("EmbeddingGenerator is being dropped");
        // Make sure the child process is terminated
        if let Some(mut child) = self.child_process.take() {
            info!("Terminating child process in Drop");
            let _ = child.kill();
            // Wait for the process to exit
            match child.wait() {
                Ok(status) => {
                    info!("Child process exited with status: {:?}", status);
                }
                Err(e) => {
                    info!("Failed to wait for child process: {}", e);
                }
            }
        }
        info!("EmbeddingGenerator drop completed");
    }
}