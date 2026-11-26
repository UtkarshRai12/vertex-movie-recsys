# Build a Vector Search index over the movie embeddings and query one user.
# Index build and deploy each take tens of minutes. The deployed index bills
# per node-hour until it is undeployed. This script leaves it deployed.
# Run from the repo root: python -m serving.vector_index
import json
from pathlib import Path

from google.cloud import aiplatform, storage
from google.cloud.aiplatform.matching_engine.matching_engine_index_config import (
    DistanceMeasureType,
)

import config

# Saved so later phases can reuse the endpoint without reading the console.
ENDPOINT_PATH = Path(__file__).with_name("index_endpoint.json")
# Must start with a letter. Letters, numbers, and underscores only.
DEPLOYED_INDEX_ID = "movies_v1"


def main() -> None:
    aiplatform.init(project=config.PROJECT, location=config.REGION)

    # Tree-AH is the approximate index. Dot product matches cosine here
    # because the two-tower export L2-normalised every vector.
    # The SDK omits algorithmConfig unless a leaf setting is passed, and
    # the API rejects the index without it. 1000 and 10 are the service defaults.
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="movies-index",
        contents_delta_uri=f"{config.BUCKET}/emb/movies",
        dimensions=64,  # best dim from Phase 3
        approximate_neighbors_count=50,
        leaf_node_embedding_count=1000,
        leaf_nodes_to_search_percent=10,
        distance_measure_type=DistanceMeasureType.DOT_PRODUCT_DISTANCE,
    )
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name="movies-endpoint",
        public_endpoint_enabled=True,
    )
    # One replica is enough for this small index. The SDK default is 2.
    endpoint.deploy_index(
        index=index,
        deployed_index_id=DEPLOYED_INDEX_ID,
        min_replica_count=1,
        max_replica_count=1,
    )

    record = {
        "index": index.resource_name,
        "index_endpoint": endpoint.resource_name,
        "deployed_index_id": DEPLOYED_INDEX_ID,
    }
    ENDPOINT_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print("INDEX_ENDPOINT =", endpoint.resource_name)

    # First line of the user export is the query. User 1 in the current file.
    line = (
        storage.Client(project=config.PROJECT)
        .bucket(config.BUCKET[5:])
        .blob("emb/users/users.json")
        .download_as_text()
        .splitlines()[0]
    )
    user = json.loads(line)
    neighbors = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[user["embedding"]],
        num_neighbors=10,
    )
    print(user["user_id"], [(n.id, round(n.distance, 3)) for n in neighbors[0]])


if __name__ == "__main__":
    main()
