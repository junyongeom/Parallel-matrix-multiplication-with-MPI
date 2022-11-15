#define _GNU_SOURCE

#include <immintrin.h>
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct thread_arg {
  const float *A;
  float *transB;
  float *C;
  int N;
  int K;
  int rank; // process id
  int start;
  int end;
  int num_threads;
} args[256];
static pthread_t threads[256];

static void *matmul_kernel(void *arg) {
  struct thread_arg *my_arg = (struct thread_arg*) arg;
  const float* A = my_arg->A;
  float* B = my_arg->transB;
  float* C = my_arg->C;
  int N = my_arg->N;
  int K = my_arg->K;
  int my_id = my_arg->rank;
  int jstart = my_arg->start + my_id;
  int jend = my_arg->end;
  int num_threads = my_arg->num_threads;
  
  for(int idx = jstart; idx <jend; idx+=num_threads){
    int aidx = (idx/N) * K;  
    int bidx = (idx%N) * K;
    int k = 0;    
    __m256 sum;
    sum = _mm256_set1_ps(0.0f);   
    for(; k+8 <= K; k+=8){  // k+8 <= K;
      __m256 l = _mm256_loadu_ps(&A[aidx+k]); // must use loadu not b 
      __m256 r = _mm256_loadu_ps(&B[bidx+k]); // because of the memory alignment matter
      sum = _mm256_fmadd_ps(l, r, sum);   
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_extractf128_ps(sum, 0);
    lo = _mm_add_ps(hi, lo);
    hi = _mm_movehl_ps(hi, lo);
    lo = _mm_add_ps(hi, lo);
    hi = _mm_shuffle_ps(lo, lo, 1);
    lo = _mm_add_ss(hi, lo);
    C[idx] = _mm_cvtss_f32(lo);
    for (; k < K; k++){
      C[idx] += A[aidx+k] * B[bidx+k];
    }    
  }   
  
  return NULL;
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int threads_per_process, int mpi_rank, int mpi_world_size) { 
  int err;
  MPI_Status status;

  if (threads_per_process > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }
  int JPN = (M*N) / mpi_world_size; // Jobs per node
  float* transposeB;
  transposeB = (float*)malloc(sizeof(float)*(N*K)); 
  int offset = M*N - JPN * (mpi_world_size-1); // root node is responsible for taking remaining jobs
  int start, end;

  if(mpi_rank==0){ //root  
    start = 0;
    end = offset; 
    
    /* transpose matrix B*/
    for (int i = 0; i < K; ++i){
      for (int j = 0; j < N; ++j) {
        transposeB[i+K*j] = B[i*N+j];
      }
    }
    
    /* distributes matrices */
    for(int i = 1; i < mpi_world_size; i++){
      MPI_Send(A, M*K, MPI_FLOAT, i, 1000, MPI_COMM_WORLD);
      MPI_Send(transposeB, N*K, MPI_FLOAT, i, 1001, MPI_COMM_WORLD);
    }
    
    /* thread level parallel execution */
    for (int t = 0; t < threads_per_process; t++){
      args[t].A = A, args[t].transB = transposeB, args[t].C = C, args[t].K = K, args[t].N = N,
      args[t].rank = t, args[t].start = start, args[t].end = end, args[t].num_threads = threads_per_process;
      err = pthread_create(&threads[t], NULL, matmul_kernel, (void *)&args[t]);
      if (err) {
        printf("pthread_create(%d) failed with err %d\n", t, err);
        exit(EXIT_FAILURE);
      }
    }
    for (int t = 0; t < threads_per_process; ++t) {
      err = pthread_join(threads[t], NULL);
      if (err) {
        printf("pthread_join(%d) failed with err %d\n", t, err);
        exit(EXIT_FAILURE);
      }
    }
    
    /* gathers matrices (vector) */
    for(int i = 1; i < mpi_world_size; i++){
      MPI_Recv(&C[offset+JPN*(i-1)], JPN, MPI_FLOAT, i, 1002, MPI_COMM_WORLD,&status);
    }
    free(transposeB); 
  }

  else{ // child node
    int start = offset + (mpi_rank-1) * JPN;
    int end = start + JPN;
    float* newA;
    newA = (float*)malloc(sizeof(float)*(M*K)); 
    
    /* gets matrices */
    MPI_Recv(newA, M*K, MPI_FLOAT, 0, 1000, MPI_COMM_WORLD,&status);
    MPI_Recv(transposeB, N*K, MPI_FLOAT, 0, 1001, MPI_COMM_WORLD,&status);
    
    /* thread level parallel execution */
    for (int t = 0; t < threads_per_process; t++){
      args[t].A = newA, args[t].transB = transposeB, args[t].C = C, args[t].K = K, args[t].N = N,
      args[t].rank = t, args[t].start = start, args[t].end = end, args[t].num_threads = threads_per_process;
      err = pthread_create(&threads[t], NULL, matmul_kernel, (void *)&args[t]);
      if (err) {
        printf("pthread_create(%d) failed with err %d\n", t, err);
        exit(EXIT_FAILURE);
      }
    }
    for (int t = 0; t < threads_per_process; ++t) {
      err = pthread_join(threads[t], NULL);
      if (err) {
        printf("pthread_join(%d) failed with err %d\n", t, err);
        exit(EXIT_FAILURE);
      }
    }
    
    /* returns calculation results */
    MPI_Send(&C[offset+JPN*(mpi_rank-1)], JPN, MPI_FLOAT, 0, 1002, MPI_COMM_WORLD);
    
    free(newA);
    free(transposeB); 
  }
  MPI_Barrier(MPI_COMM_WORLD);   
}
