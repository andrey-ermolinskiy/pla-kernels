# pla-kernels
A collection of high-performance parallel linear algebra kernels for shared-memory multiprocessors. These kernels are implemented in C++ and rely on OpenMP for parallelization.

## Kernels
### SpMSpV
Sparse matrix-sparse vector (SpMSpV) multiplication of the form y = Ax is a widely used computational kernel with many applications in machine learning and graph analytics. A sparse input matrix A is multiplied by a sparse input vector x to produce a sparse output vector y.

**src/spmspv/** implements the SpMSpV-bucket algorithm that was recently proposed by Azad and Buluc [[1]](#1). Unlike most prior approaches, this algorithm achieves *work-efficiency*, which means that the total work performed by all processors remains within a constant factor of the fastest known serial algorithm.

The 

## References
<a id="1">[1]</a> 
Ariful Azad, Aydin Buluc.
*A Work-Efficient Parallel Sparse Matrix-Sparse Vector Multiplication Algorithm.*
IPDPS 2017: 688-697.





<img src="eval/spmspv/runtime-ljournal-2008.png" width="60%" height="60%">
<img src="eval/spmspv/runtime-hugetrace-00020.png" width="60%" height="60%"> 

