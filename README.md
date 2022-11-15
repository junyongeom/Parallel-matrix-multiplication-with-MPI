# Parallel-matrix-multiplication-with-MPI

When multiplying matrices of size M\*K (matrix A) and K\*N (matrix B), it takes K multiplications (and sum) M\*N times.  
So, I first set the number of total jobs as M\*N, and divided it with the number of working nodes size (MPI).  
The root node is responsible for taking remaining jobs that can not divided by the number of nodes.

And then, each thread performs matrix (rather vector) multiplication in parallel for the task assigned to the corresponding node.

Furthermore, to use AVX2 instructions, I transpose the matrix B before parallel multiplication.  
Because to use the AVX2 fmadd, the addresses of the variables to be multiplied and added must be in order.  

As a result, I could parallelized the matrix multiplication with the node (MPI), thread (pthread), and operation (AVX2) levels.
