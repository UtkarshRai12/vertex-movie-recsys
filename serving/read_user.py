# Read one user back from the online feature store.
# Run from the repo root: python -m serving.read_user
import vertexai
from vertexai.resources.preview.feature_store import FeatureView

import config

vertexai.init(project=config.PROJECT, location=config.REGION)
view = FeatureView("user_features", feature_online_store_id="recsys_online")
print(view.read(key=["1"]).to_dict())
