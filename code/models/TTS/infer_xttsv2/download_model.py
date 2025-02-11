'''
download checkpoints from huggingface
'''

from huggingface_hub import snapshot_download
import os

# Specify your target directory
target_dir = "checkpoints"

# Create the directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Download all files from the model
snapshot_download(
    repo_id="coqui/XTTS-v2",
    local_dir=target_dir,
    local_dir_use_symlinks=False  # Set to True if you want to use symlinks instead of copying files
)

print(f"All files downloaded to {target_dir}")