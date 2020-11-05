docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
-v ~/Documents/coding_projects/headroom/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:20.10-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose 1
