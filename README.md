I downloaded the server using 

```
docker pull nvcr.io/nvidia/tritonserver:20.10-py3
```

I created the model in `blazeface/1/model.pt` by running the `main.py` code from inside the `MediaPipePyTorch` repo.

I run the server using the provided `runit.sh` script.

For testing my server I split the webm into jpeg frames by 
```
mkdir frames
python3 splitwebm.py
```

Then trying to call the server using
```
python3 image_client.py -v -i grpc -u localhost:8001 -m blazeface -s NONE frames/frame0.jpg
```
but this doesn't work.
It ends with the error 
```
I1105 02:42:27.139824 1 libtorch_backend.cc:776] isTuple() INTERNAL ASSERT FAILED at "/opt/tritonserver/include/torch/ATen/core/ivalue_inl.h":931, please report a bug to PyTorch. Expected Tuple but got GenericList
```

Calling a different model 
```
python3 image_client.py -i grpc -u localhost:8001 -b 128 -m inception_graphdef -s INCEPTION frames > results.txt
```
works successfully and the results are in results.txt. Tarrance looks most like a potters wheel or a banjo.