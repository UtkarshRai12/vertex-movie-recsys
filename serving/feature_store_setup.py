# Create the online feature store and copy user_serving into it.
# The store bills for as long as it exists. The sync takes a few minutes.
# Run from the repo root: python -m serving.feature_store_setup
import json
import time
from pathlib import Path

import vertexai
from vertexai.resources.preview.feature_store import FeatureOnlineStore
from vertexai.resources.preview.feature_store.utils import FeatureViewBigQuerySource

import config

STORE_NAME = "recsys_online"
VIEW_NAME = "user_features"
RECORD_PATH = Path(__file__).with_name("feature_store.json")


def wait_for_sync(sync, timeout_s: int = 20 * 60) -> None:
    # sync() only submits the copy. Poll until the run has an end time.
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # gca_resource is cached. Refresh it or the end time never appears.
        sync._sync_gca_resource()
        resource = sync.gca_resource
        end = resource.run_time.end_time
        if end.seconds or end.nanos:
            code = resource.final_status.code
            rows = resource.sync_summary.row_synced
            print(f"sync finished code={code} rows={rows}")
            if code:
                raise SystemExit(resource.final_status.message or f"sync failed: {code}")
            return
        print("sync still running")
        time.sleep(20)
    raise SystemExit("sync did not finish within 20 minutes")


def main() -> None:
    vertexai.init(project=config.PROJECT, location=config.REGION)

    # Optimized stores are rejected by the API now. Bigtable is the
    # supported online store, and FeatureView.read works the same way.
    # One node is enough for this dataset. It bills until the store is deleted.
    store = FeatureOnlineStore.create_bigtable_store(
        STORE_NAME,
        min_node_count=1,
        max_node_count=1,
    )
    view = store.create_feature_view(
        VIEW_NAME,
        source=FeatureViewBigQuerySource(
            uri=f"bq://{config.PROJECT}.movielens.user_serving",
            entity_id_columns=["user_id"],
        ),
    )
    sync = view.sync()
    print("store:", store.resource_name)
    print("view:", view.resource_name)
    print("sync:", sync.resource_name)
    wait_for_sync(sync)

    record = {
        "store": store.resource_name,
        "view": view.resource_name,
        "store_id": STORE_NAME,
        "view_id": VIEW_NAME,
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
