# TorchServe CPP (Experimental Release)
## Build & Unit tests
### Requirements
* C++17
* GCC version: gcc-9
### Commands
```
cd serve/cpp 
./build.sh
```
## Backend
TorchServe cpp backend can run as a process, which is similar to [TorchServe Python backend](https://github.com/pytorch/serve/tree/master/ts). By default, TorchServe supports torch scripted model in cpp backend. [src/backends/core/backend.hh](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/backends/core/backend.hh) defines the APIs of backend to support multiple different platforms such as MxNet, ONNX and so on. 
* [Backend](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/backends/core/backend.hh#L60) defines function `LoadModelInternal` to support model loading on different platforms.
* [ModelInstance](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/backends/core/backend.hh#L25) represents a model copy. The function `Predict` is to support prediction on different platforms.
### TorchScripted Backend
By default, TorchServe cpp provides [TorchScripted backend](https://github.com/pytorch/serve/tree/cpp_backend/cpp/src/backends/torch_scripted). Its [basd handler](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/backends/torch_scripted/handler/base_handler.hh) defines APIs to customize handler.
* [Initialize](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/backends/torch_scripted/handler/base_handler.hh#L29)
* [LoadModel](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/backends/torch_scripted/handler/base_handler.hh#L37)
* [Preprocess](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/backends/torch_scripted/handler/base_handler.hh#L40)
* [Inference](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/backends/torch_scripted/handler/base_handler.hh#L46)
* [Postprocess](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/backends/torch_scripted/handler/base_handler.hh#L53)
#### Example
##### Using basehandler
* set runtime as "LSP" in model archiver option [--runtime](https://github.com/pytorch/serve/tree/master/model-archiver#arguments) 
* set handler as "BaseHandler" in model archiver option [--handler](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
```
 torch-model-archiver --model-name mnist_base --version 1.0 --serialized-file mnist_script.pt --handler BaseHandler --runtime LSP
```
Here is an [example](https://github.com/pytorch/serve/tree/cpp_backend/cpp/test/resources/torchscript_model/mnist/base_handler) of unzipped model mar file.
##### Using customized handler
* build customized handler shared lib. For example [Mnist handler](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/examples/image_classifier/mnist).
* set runtime as "LSP" in model archiver option [--runtime](https://github.com/pytorch/serve/tree/master/model-archiver#arguments) 
* set handler as "libmnist_handler:MnistHandler" in model archiver option [--handler](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
* set --extra-files as "libmnist_handler.dylib" in model archiver option [--extra-files](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
```
torch-model-archiver --model-name mnist_base --version 1.0 --serialized-file mnist_script.pt --handler libmnist_handler:MnistHandler --runtime LSP
```
Here is an [example](https://github.com/pytorch/serve/tree/cpp_backend/cpp/test/resources/torchscript_model/mnist/mnist_handler) of unzipped model mar file.
##### Mnist example
* Transform data on client side. For example:
```
import torch
from PIL import Image
from torchvision import transforms

image_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
image = Image.open("examples/image_classifier/mnist/test_data/0.png")
image = image_processing(image)
torch.save(image, "0_png.pt")
```
* Run model registration and prediction: [Using basehandler](https://github.com/pytorch/serve/blob/cpp_backend/cpp/test/backends/torch_scripted/torch_scripted_backend_test.cc#L54) or [Using customized handler](https://github.com/pytorch/serve/blob/cpp_backend/cpp/test/backends/torch_scripted/torch_scripted_backend_test.cc#L72).

