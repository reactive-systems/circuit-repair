"""Google Cloud Platform storage bucket utility"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from google.cloud import storage
from tqdm import tqdm
import pandas as pd
import json
import argparse

from .globals import ML2_BUCKET


def latest_version(bucket_dir: str, name: str, bucket_name: str = ML2_BUCKET):
    storage_client = storage.Client()
    latest_version = -1
    prefix = f"{bucket_dir}/{name}-"
    for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
        _, suffix = blob.name.split(prefix, 1)
        version_str, _ = suffix.split("/", 1)
        if version_str.isdigit() and int(version_str) > latest_version:
            latest_version = int(version_str)
    return latest_version


def path_exists(path: str, bucket_name: str = ML2_BUCKET):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=path)
    return any(True for _ in blobs)


def upload_path(local_path: str, bucket_path: str, bucket_name: str = ML2_BUCKET):
    """Uploads a file or directory to a GCP storage bucket

    Args:
        local_path: path to a local file or directory
        bucket_path: path where the file or directory is stored in the bucket
        bucket_name: name of the GCP storage bucket

    Raises:
        Exception: if the local path is not a valid path to a file or directory
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if os.path.isfile(local_path):
        blob = bucket.blob(bucket_path)
        blob.upload_from_filename(local_path)
    elif os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            bucket_root = root.replace(local_path, bucket_path, 1)
            for file in files:
                if file.startswith("."):
                    # skip hidden files
                    continue
                blob = bucket.blob(f"{bucket_root}/{file}")
                blob.upload_from_filename(f"{root}/{file}")
            for d in dirs:
                if d.startswith("."):
                    # skip hidden directories
                    dirs.remove(d)
    else:
        raise Exception("Path %s is not a valid path to a file or directory", local_path)


def download_file(filename: str, local_filename: str, bucket_name: str = ML2_BUCKET):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.download_to_filename(local_filename)


def download_path(
    bucket_path: str, local_path: str, recurse: bool = True, bucket_name: str = ML2_BUCKET
):
    """Downloads a directory from a GCP storage bucket

    Args:
        bucket_path: path that identifies a file in the bucket or a prefix that emulates a directory in the bucket
        local_path: path where the file or directory is stored locally
        recurse: whether to recurse on a directory
        bucket_name: name of the GCP storage bucket
    """
    storage_client = storage.Client()
    delimiter = None if recurse else "/"
    blobs = storage_client.list_blobs(bucket_name, prefix=bucket_path + "/", delimiter=delimiter)
    for blob in tqdm(blobs):
        filepath = blob.name.replace(bucket_path, local_path, 1)
        file_dir, filename = os.path.split(filepath)
        if not filename:
            # if filename is empty blob is a folder
            continue
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        blob.download_to_filename(filepath)


def download_metadata_all(
    artifact_prefix: str, project_dir: str, bucket_name: str = ML2_BUCKET
) -> pd.DataFrame:
    """Constructs a dataframe containing all metadata / arguments from a project with a given artifact prefix.

    Args:
        artifact_prefix (str): All artifacts, which metadata is collected must begin with the given prefix.
        project_dir (str): the project directory where the artifacts lie in (i.e. ltl-syn)
        bucket_name (str, optional): The google cloud bucket name. Defaults to the content of global variable ML2_BUCKET.
    """

    def flatten(obj: dict, parent_string: str = "") -> dict:
        """Transforms any given dict with lists or dicts as elements to a dict with just a single depth, hence all values in this dicts are basic datatypes.

        Example:
            {
                "a": [1,2,3,4],
                "b": {
                    "c": 1,
                    "d": 2,
                }
            }
            becomes
            {
                "a.0": 1,
                "a.1": 2,
                "a.2": 3,
                "a.3": 4,
                "b.c": 1,
                "b.d": 2
            }

        Args:
            obj (dict): input dict
            parent_string (str, optional): For recursive remembering of the parent keys. Defaults to ""

        Returns:
            dict: the flattened dict
        """
        new_obj = {}
        for k in obj:
            if isinstance(obj[k], list):
                for i, v in enumerate(obj[k]):
                    new_obj[parent_string + k + "." + str(i)] = v
            elif isinstance(obj[k], dict):
                new_obj = {**new_obj, **flatten(obj[k], parent_string + k + ".")}
            else:
                new_obj[parent_string + k] = obj[k]
        return new_obj

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=project_dir + "/" + artifact_prefix)
    series = []
    for blob in blobs:
        if blob.name.endswith("/metadata.json"):
            artifact = blob.name[len(project_dir + "/") : blob.name.find("/metadata.json")]
            series.append(pd.Series(flatten(json.loads(blob.download_as_text())), name=artifact))
        if blob.name.endswith("/args.json"):
            artifact = blob.name[len(project_dir + "/") : blob.name.find("/args.json")]
            series.append(pd.Series(flatten(json.loads(blob.download_as_text())), name=artifact))

    dl = pd.concat(series, axis=1).T
    dl.reindex(sorted(dl.columns), axis=1)
    dl.sort_index()
    return dl


def find(
    project_dir: str,
    datasets_metadata: Optional[Dict] = None,
    experiments_args: Optional[Dict] = None,
    name_includes: List[str] = [],
    name_excludes: List[str] = [],
    ignore_missing_keys: bool = True,
    bucket: str = ML2_BUCKET,
    return_param: List[str] = None,
) -> List[Union[str, Tuple[str, Any]]]:
    """Find all datasets in google cloud according to the given parameter (metadata/args).

    The arguments datasets_metadata and experiments_args specify filters. If not set, no datasets resp. experiments are collected. If empty, not filter is set, Hence all datasets resp. experiments are collected.

    The dictionary should have the same structure as the metadata / args file, including hierarchy. Instead of a value a tuple is expected, configuring whether we search for values equal the filter value (=), greater the filter value (>), less (<), greater equal (>=) or less equal (<=). Make sure types match and are comparable, otherwise it will throw an exception.

    Args:
        project_dir (str): The project directory (such as ltl-syn)
        datasets_metadata (Optional[Dict], optional): Filter options for datasets. If not set, no datasets are collected. Defaults to None.
        experiments_args (Optional[Dict], optional): Filter options for experiments. If not set, no experiments are collected. Defaults to None.
        name_includes (List[str], optional): A list of strings that should be included in the blob path. Defaults to [].
        name_excludes (List[str], optional): A list of strings that must not be included in the blob path. Defaults to [].
        ignore_missing_keys (bool, optional): If True, keys specified in the filters do not need to exist in the cloud dataset. Defaults to True.
        bucket (str, optional): The google cloud bucket. Defaults to ML2_BUCKET.
        return_param (List[str], optional): If set, we return a parameter from the artifact metadata together with the name of the artifact. Specify a list of hierachical keys to reach the parameter you want. Including empty list, for all data.

    Returns:
        List[str]: A list of all datasets/experiments that match the criteria
    """

    def check_name(name: str, includes: List[str], excludes: List[str]) -> bool:
        for include in includes:
            if name.find(include) == -1:
                return False
        for exclude in excludes:
            if name.find(exclude) != -1:
                return False
        return True

    def check_filter(blob_dict: Dict, filter: Dict) -> bool:
        for k, v in filter.items():
            try:
                if blob_dict[k] is None:
                    if ignore_missing_keys:
                        print("Key " + str(k) + " is None in " + blob.name + ". Ignore...")
                        continue
                    else:
                        return False
            except KeyError:
                if ignore_missing_keys:
                    print("Key " + str(k) + " not found in " + blob.name + ". Ignore...")
                    continue
                else:
                    return False
            if isinstance(v, tuple):
                value, operator = v
                if operator == "=":
                    if blob_dict[k] != value:
                        return False
                elif operator == ">":
                    if blob_dict[k] <= value:
                        return False
                elif operator == "<":
                    if blob_dict[k] >= value:
                        return False
                elif operator == ">=":
                    if blob_dict[k] < value:
                        return False
                elif operator == "<=":
                    if blob_dict[k] > value:
                        return False
                else:
                    raise TypeError(
                        str(v)
                        + " has not a correct type, "
                        + operator
                        + " is not a valid operator"
                    )
            elif isinstance(v, Dict):
                if not check_filter(blob_dict=blob_dict[k], filter=v):
                    return False
            elif isinstance(v, List):
                raise NotImplementedError("Lists not supported in filter")
            else:
                raise TypeError(
                    str(v)
                    + " has not a correct type, use a tuple with (..., =) to express equality"
                )
        return True

    def handle_return_param(artifact_data: Dict):
        try:
            res = artifact_data
            for el in return_param:
                res = res[el]
            return res
        except KeyError:
            return None

    def handle_artifact(
        blob, filename: str, filter: Dict
    ) -> Optional[Union[str, Tuple[str, Any]]]:
        artifact_data = json.loads(blob.download_as_text())
        if check_filter(artifact_data, filter):
            blob_name = blob.name[
                blob.name.find(project_dir + "/") + len(project_dir) + 1 : blob.name.find(filename)
            ]
            if return_param is None:
                return blob_name
            else:
                return blob_name, handle_return_param(artifact_data)

        return None

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket, prefix=project_dir + "/")
    data = []
    for blob in blobs:
        if check_name(blob.name, includes=name_includes, excludes=name_excludes):
            if datasets_metadata is not None and blob.name.endswith("/metadata.json"):
                res = handle_artifact(
                    blob=blob, filter=datasets_metadata, filename="/metadata.json"
                )
                if res is not None:
                    data.append(res)
            if experiments_args is not None and blob.name.endswith("/args.json"):
                res = handle_artifact(blob=blob, filter=experiments_args, filename="/args.json")
                if res is not None:
                    data.append(res)
    return data


def get_metadata(artifact_name: str, project_dir: str, bucket_name: str = ML2_BUCKET) -> Dict:
    """Returns the metadata/ args file for a artifact.

    Args:
        artifact_name (str): the artifact name
        project_dir (str): In which project the artifact lies
        bucket_name (str, optional): The bucket of the artifact. Defaults to ML2_BUCKET.

    Raises:
        FileNotFoundError: If nor metadata / args file found

    Returns:
        Dict: The metadata / args for this artifact
    """
    filename = project_dir + "/" + artifact_name + "/"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename + "args.json")
    if blob.exists():
        return json.loads(blob.download_as_text())
    blob = bucket.blob(filename + "metadata.json")
    if blob.exists():
        return json.loads(blob.download_as_text())
    raise FileNotFoundError()


def cli() -> None:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--artifact-prefix",
        type=str,
        dest="artifact_prefix",
        default="None",
        help="All artifacts, which metadata /arguments is collected must begin with this prefix.",
    )
    dir = parser.add_argument(
        "-p",
        "--project-dir",
        type=str,
        dest="project_dir",
        default=None,
        help="The project directory where the artifacts lie in (i.e. ltl-syn)",
    )
    parser.add_argument(
        "-o", "--output", type=str, dest="output", default=None, help="output path"
    )
    parser.add_argument(
        "-w",
        "--add-to-wandb",
        dest="wandb",
        action="store_true",
        default=False,
        help="Add table to Weights and Biases",
    )

    args = parser.parse_args()
    args_dict = vars(args)

    if not args_dict["project_dir"]:
        raise argparse.ArgumentError(
            message="You need to support a project from which you want to download (i.e. ltl-syn)",
            argument=dir,
        )

    logger.info(
        "Collecting Metadata from "
        + args_dict["project_dir"]
        + "/"
        + args_dict["artifact_prefix"]
        + "..."
    )
    dataframe = download_metadata_all(args_dict["artifact_prefix"], args_dict["project_dir"])

    if args_dict["output"]:
        dataframe.to_csv(args_dict["output"])
        logger.info("Saved to " + args_dict["output"])

    if args_dict["wandb"]:
        import wandb

        run = wandb.init(project=args_dict["project_dir"])
        my_table = wandb.Table(dataframe=dataframe)
        run.log({"Artifacts with prefix " + args_dict["artifact_prefix"]: my_table})
        logger.info("Uploaded to wandb")


if __name__ == "__main__":
    cli()
