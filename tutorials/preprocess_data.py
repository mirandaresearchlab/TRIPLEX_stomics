# trying to fix tqdm going wild
import sys, builtins, functools, tqdm
_original = tqdm.tqdm                 # save first
tqdm.tqdm = functools.partial(
    _original,
    file=sys.stderr,
    dynamic_ncols=True
)
builtins.print = lambda *a, **k: _original.write(" ".join(map(str, a)), **k)


import os
if os.path.basename(os.getcwd()) == 'tutorials':
    # Change to parent directory
    os.chdir('..')


from src.preprocess.pipeline import TriplexPipeline, get_config

# Since their pipeline need a csv with the ids-patient mapping, we create one before running the pipeline
import pathlib
import pandas as pd
import hest
import uuid

from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))  # Login to Hugging Face Hub



def main():
        
    hest_package_path = pathlib.Path(hest.__file__).resolve()
    project_root = hest_package_path.parents[2]
    csv_path = project_root / "assets" / "HEST_v1_1_0.csv"


    meta = pd.read_csv(csv_path)

    def get_downloaded_sample_ids(path: pathlib.Path) -> list:
        """
        Get the list of sample IDs from the downloaded HEST data.
        """
        return [f.stem for f in path.glob('*.tif') if f.is_file()]
    ids = get_downloaded_sample_ids(pathlib.Path("/storage/hest_data/hs_brain/wsis"))

    ids_df = meta[meta["id"].isin(ids)][["id", "patient"]]

    # fill NaN and empty patient values with a unique identifier
    ids_df["patient"] = ids_df["patient"].replace(r"^\s*$", pd.NA, regex=True)  # normalize empty strings to NaN
    mask = ids_df["patient"].isna()
    n_missing = mask.sum()
    if n_missing:
        ids_df.loc[mask, "patient"] = [uuid.uuid4().hex for _ in range(n_missing)]

    # rename id column to sample_id
    ids_df.rename(columns={"id": "sample_id"}, inplace=True)

    ids_df.to_csv("/storage/hest_data/hs_brain/ids.csv", index=False)
    ids_df.to_csv("/storage/hest_data/hs_brain/processed_data_triplex/ids.csv", index=False)


    # Example of processing HEST data
    hest_config = {
        # Basic configuration
        'input_dir': '/storage/hest_data/hs_brain',
        'output_dir': '/storage/hest_data/hs_brain/processed_data_triplex',
        'mode': 'hest',
        
        # Preprocessing parameters
        'hest_dir': '/storage/hest_data/hs_brain',
        'slide_ext': '.tif',
        'save_neighbors': True,
        'n_splits': 2,
        'n_top_hvg': 50,
        'n_top_heg': 1000,
        'n_top_hmhvg': 200,
        
        # Feature extraction parameters
        'model_name': 'uni_v1',
        'batch_size': 256,
        'num_workers': 1,
        'feature_type': 'both',
        'gpus': [0]
    }

    pipeline = TriplexPipeline(hest_config)
    pipeline.run_pipeline()  # This will run preprocessing and feature extraction


def _cli_entrypoint():
    main()

if __name__ == "__main__":
    _cli_entrypoint()
