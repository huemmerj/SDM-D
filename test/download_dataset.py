import kagglehub

# Download latest version
path = kagglehub.dataset_download("mmwang0/megafruits")

print("Path to dataset files:", path)