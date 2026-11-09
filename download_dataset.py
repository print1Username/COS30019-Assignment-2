from pathlib import Path

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="print1Username/COS30019-Assignment-2.dataset",
    repo_type="dataset",
    allow_patterns="dataset/**",
    local_dir=Path(__file__).resolve().parent,
)
