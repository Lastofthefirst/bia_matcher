use qdrant_client::qdrant::{PointStruct, VectorParamsBuilder, Distance, CreateCollectionBuilder, UpsertPointsBuilder};
use qdrant_client::{Payload, Qdrant};
use ndarray::{Array, Array2};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Qdrant::from_url("http://localhost:6334").build()?;

    let collection_name = "test_collection";
    
    // Delete collection if it exists
    let _ = client.delete_collection(collection_name).await;
    
    // Create collection
    client
        .create_collection(
            CreateCollectionBuilder::new(collection_name)
                .vectors_config(VectorParamsBuilder::new(256, Distance::Dot)),
        )
        .await?;

    // Create a point with payload - simulating our main application
    let payload: Payload = serde_json::json!({
        "doc_id": "test_doc",
        "element_id": "test_element",
        "text": "This is a test text"
    }).try_into().unwrap();

    // Create a vector - simulating our main application
    let vector: Vec<f32> = vec![0.1; 256];

    let points = vec![PointStruct::new(
        1,
        vector,
        payload,
    )];

    // Upsert points
    client
        .upsert_points(UpsertPointsBuilder::new(collection_name, points).wait(true))
        .await?;

    println!("Successfully created test point with 256-dimensional vector");
    
    Ok(())
}