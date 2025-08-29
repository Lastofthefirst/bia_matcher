// Test file for enhanced document matching approach

use matching::models::TextStatistics;

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

fn calculate_composite_score(semantic: f32, length: f32, word: f32) -> f32 {
    0.7 * semantic + 0.15 * length + 0.15 * word
}

fn normalize_filename(filename: &str) -> String {
    filename.replace("-", " ").replace("_", " ")
}

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

fn main() {
    println!("=== Enhanced Document Matching Test ===\n");
    
    // Test text statistics
    let text1 = "This is a sample text with several words";
    let text2 = "This is another sample text with more words than the first one";
    
    let stats1 = TextStatistics::new(text1);
    let stats2 = TextStatistics::new(text2);
    
    println!("Text Statistics:");
    println!("Text 1: '{}'", text1);
    println!("  Character length: {}", stats1.char_length);
    println!("  Word count: {}", stats1.word_count);
    
    println!("Text 2: '{}'", text2);
    println!("  Character length: {}", stats2.char_length);
    println!("  Word count: {}", stats2.word_count);
    
    // Test length similarity
    let length_sim = stats1.length_similarity(&stats2);
    println!("Length similarity: {:.4}", length_sim);
    
    // Test word similarity
    let word_sim = stats1.word_similarity(&stats2);
    println!("Word similarity: {:.4}", word_sim);
    
    // Test cosine similarity with sample vectors
    let vec1 = vec![0.5, 0.3, 0.2, 0.1];
    let vec2 = vec![0.4, 0.4, 0.1, 0.1];
    
    let cos_sim = cosine_similarity(&vec1, &vec2);
    println!("Cosine similarity (sample vectors): {:.4}", cos_sim);
    
    // Test composite score calculation
    let semantic_sim = 0.85; // Example semantic similarity
    let composite_score = calculate_composite_score(semantic_sim, length_sim, word_sim);
    println!("Semantic similarity: {:.4}", semantic_sim);
    println!("Composite score: {:.4}", composite_score);
    
    // Test filename similarity
    let filename1 = "abdul-baha_wisdom_paris_talks";
    let filename2 = "wisdom_abdul_baha_paris_talks_processed";
    
    let norm1 = normalize_filename(filename1);
    let norm2 = normalize_filename(filename2);
    
    println!("\nFilename Similarity:");
    println!("Filename 1: '{}' -> '{}'", filename1, norm1);
    println!("Filename 2: '{}' -> '{}'", filename2, norm2);
    
    let filename_sim = filename_similarity(filename1, filename2);
    println!("Filename similarity: {:.4}", filename_sim);
    
    // Test with completely different filenames
    let filename3 = "some_other_document";
    let filename4 = "totally_different_file";
    
    let diff_filename_sim = filename_similarity(filename3, filename4);
    println!("Different filenames '{}' vs '{}': {:.4}", filename3, filename4, diff_filename_sim);
    
    println!("\n=== Test Summary ===");
    println!("The enhanced matching approach combines:");
    println!("- 70% semantic similarity (from embeddings)");
    println!("- 15% length similarity (structural feature)");
    println!("- 15% word count similarity (structural feature)");
    println!("This provides better discrimination between correct and incorrect matches.");
    println!("Filename similarity is used as a secondary factor to boost scores of");
    println!("documents with similar naming patterns.");
}