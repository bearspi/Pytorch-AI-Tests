[CheatSheet](https://pytorch.org/tutorials/beginner/ptcheat.html)

Running on MPS is about % 130 faster than CPU
 
Matrix Mult requires reversed **dimensions**:

* (3,2) @ (3,2) ❌
* (3,2) @ (2,3) ✅
* (2,3) @ (3,2) ✅

Matrix Mult has the shape of the outer **dimensions**:

* (3,2) @ (2,3) -> (3,3)
* (2,3) @ (3,2) -> (2,2)

When converting `NumPy arrays` -> `PyTorch Tensors` it cames as **64 Bit** values.

Numpy and PyPlot can't work with datas on MPS (GPU).
"Loss Func" can be called "Cost Func" or "Criterion"

When making NonBinary Classer dont forget to change the y value to a `torch.LongTensor`
Dont forget to switch between eval and train mode.

Convo layers and `Adam` optimizer doesnt give great results.

FMNIST_MODEL_V5 100 epochs test acc: 93.16%
FMNIST_MODEL_V6 100 epochs test acc: 93.23%