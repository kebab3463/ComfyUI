from huggingface_hub import HfApi

api = HfApi()

# Get the list of files
files = api.list_repo_files(repo_id="ApacheOne/WAN_loRAs", repo_type="model")

# Print total count
print(f"Total number of files: {len(files)}")
