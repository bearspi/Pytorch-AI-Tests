Running on MPS is abou % 130 faster than CPU
 
Matrix Mult requires reversed *dimensions*:

(3,2) @ (3,2) ❌

(3,2) @ (2,3) ✅

(2,3) @ (3,2) ✅

Matrix Mult has the shape of the outer *dimensions*:

(3,2) @ (2,3) -> (3,3)

(2,3) @ (3,2) -> (2,2)