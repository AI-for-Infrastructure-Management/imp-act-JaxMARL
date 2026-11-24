# imp-act-JaxMARL Docker Setup

This guide explains how to build and run the Docker container for the `imp-act-JaxMARL` project.
The `DockerFile` uses the NVIDIA cudnn runtime image `nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04`
and installs the required dependencies for the project. 

## 🚀 Build the Docker Image

Build the image for x86_64 (useful for deploying on standard Linux systems). 
If you're on macOS (M1/M2), make sure you're using docker buildx to cross-compile for amd64.

This will build the image for the `linux/amd64` platform and tag it as 
`yourusername/imp-act-jaxmarl:latest`,and push it to the Docker Hub. 
Make sure to replace `yourusername` with your actual Docker Hub username.
```bash
docker buildx build \
  --platform=linux/amd64 \
  -t yourusername/imp-act-jaxmarl:latest \
  --load .
```

Once the image is built, it can be run, and the project directory is
`/workspace/imp-act-JaxMARL`. The virtual environment `.venv` in this directory,
with the pinned dependencies of `uv.lock` (GPU version), is activated by default.
```bash 
cd /workspace/imp-act-JaxMARL
```