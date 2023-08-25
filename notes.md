[CheatSheet](https://pytorch.org/tutorials/beginner/ptcheat.html)

Running on MPS is abou % 130 faster than CPU
 
Matrix Mult requires reversed **dimensions**:

* (3,2) @ (3,2) ❌
* (3,2) @ (2,3) ✅
* (2,3) @ (3,2) ✅

Matrix Mult has the shape of the outer **dimensions**:

* (3,2) @ (2,3) -> (3,3)
* (2,3) @ (3,2) -> (2,2)

When converting 'NumPy arrays' -> 'PyTorch Tensors' it cames as **64 Bit** values.

Numpy cant work on datas on MPS (GPU).