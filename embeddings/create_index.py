import pinecone

# Create a Pinecone connection
pinecone.init()

# Define the name of your index
index_name = "medical-dialog-embeddings"

# Define the dimensionality of your data points
dimensionality = 1536

# Define the Pinecone index configuration
index_config = pinecone.IndexConfig(dim=dimensionality)

# Create the Pinecone index
pinecone.create_index(index_name, index_config)

# Close the Pinecone connection
pinecone.deinit()