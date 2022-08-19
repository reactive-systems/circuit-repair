from .ltl_repair_data import LTLRepairSplitData
from typing import List
import json
from google.cloud import storage


def find_datasets() -> List[str]:
    """Collect all datasets (under the prefix scpa-repair(alter|gen) )

    Returns:
        List[str]: The list of of bucket paths of the experiments (e.g [ltl-repair/exp-repair-gen-51-0/eval/pipe-.../summary.csv])
    """
    storage_client = storage.Client()
    blobs = storage_client.list_blobs("ml2-bucket", prefix="ltl-repair" + "/" + "scpa-repair")
    data = []
    for blob in blobs:
        if blob.name.endswith("/metadata.json"):
            if blob.name.find("/scpa-repair-gen") != -1 and blob.name.find("raw") == -1:
                metadata = json.loads(blob.download_as_text())
                if metadata["remove_or_alter"] == "alter" and metadata["replace_minimal"] is False:
                    data.append(
                        blob.name[
                            blob.name.find("ltl-repair/") + 11 : blob.name.find("/metadata.json")
                        ]
                    )
            elif blob.name.find("/scpa-repair-alter") != -1:
                data.append(
                    blob.name[
                        blob.name.find("ltl-repair/") + 11 : blob.name.find("/metadata.json")
                    ]
                )
    return data


def main() -> None:
    for data_name in find_datasets():
        data = LTLRepairSplitData.load(data_name, overwrite=True)
        data.add_metadata()
        data.save(
            name=data_name,
            upload=True,
            overwrite_local=True,
            overwrite_bucket=True,
            auto_version=False,
        )


if __name__ == "__main__":
    main()
