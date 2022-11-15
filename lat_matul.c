#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct thread_arg {
  const float *A;
  const float *B;
  float *C;
  float *transB;
  int M;
  int N;
  int K;
  int num_threads;
  int rank; /* id of this thread */
  int m_rank;
  int m_size;
} args[256];
static pthread_t threads[256];

static void *matmul_kernel(void *arg) {
  struct thread_arg *my_arg = (struct thread_arg*) arg;
  int M = my_arg->M;
  int N = my_arg->N;
  int K = my_arg->K;
  int m_rank = my_arg->m_rank;
  int m_size = my_arg->m_size;
  int num_threads = my_arg->num_threads;
  int my_id = my_arg->rank;
  int JPN = (M*N) / m_size; // Jobs per node
  int jstart = JPN * m_rank + my_id;
  int jend = jstart + JPN;
  if(m_rank == (m_size-1)) jend = M*N; // makes last node to work for the remaining jobs

  for(int idx = jstart; idx <jend; idx+=num_threads){
    int aidx = (idx/N) * K;  
    int bidx = (idx%N) * K;
    int k = 0;    
    __m256 sum;
    sum = _mm256_set1_ps(0.0f);
   
    for(; k+8 <= K; k+=8){  // k+8 <= K;
      __m256 l = _mm256_loadu_ps(&my_arg->A[aidx+k]);
      __m256 r = _mm256_loadu_ps(&my_arg->transB[bidx+k]);
      sum = _mm256_fmadd_ps(l, r, sum);   
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_extractf128_ps(sum, 0);
    lo = _mm_add_ps(hi, lo);
    hi = _mm_movehl_ps(hi, lo);
    lo = _mm_add_ps(hi, lo);
    hi = _mm_shuffle_ps(lo, lo, 1);
    lo = _mm_add_ss(hi, lo);
    my_arg->C[idx] = _mm_cvtss_f32(lo);
    for (; k < K; k++){
      my_arg->C[idx] += my_arg->A[aidx+k] * my_arg->transB[bidx+k];
    }    
  }   
  
  return NULL;
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int threads_per_process, int mpi_rank, int mpi_world_size) { 
  int err;

  if (threads_per_process > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }

  float* transposeB;
  transposeB = (float*)malloc(sizeof(float)*(N*K)); 
  for (int i = 0; i < K; ++i){
    for (int j = 0; j < N; ++j) {
      transposeB[i+K*j] = B[i*N+j];
    }
  }
  
  for (int t = 0; t < threads_per_process; t++){
    args[t].A = A, args[t].B = B, args[t].C = C, args[t].M = M, args[t].N = N,
    args[t].K = K, args[t].num_threads = threads_per_process, args[t].rank = t;
    args[t].m_rank = mpi_rank, args[t].m_size = mpi_world_size, args[t].transB = transposeB;
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
  
  // MPI_Status status;
  
  // int m_size = mpi_world_size;
  // int JPN = M*N / m_size;
  // int jstart = JPN * mpi_rank;
  // int jend = jstart + JPN;
  // if(mpi_rank == (m_size-1)) jend = M*N; // last node works for remaining jobs for simplicity
  
  // MPI_Barrier(MPI_COMM_WORLD);
  // if(mpi_rank==0){
  //   for(int i = 1; i < m_size-1; i++){
  //       MPI_Recv(&C[JPN * i], JPN, MPI_FLOAT, i, 1000, MPI_COMM_WORLD,&status);
  //       printf("received %d\n", i);
  //   }
  //   if(m_size!=1){
  //       MPI_Recv(&C[JPN*m_size], JPN+((M*N)%m_size), MPI_FLOAT, m_size-1, 1000, MPI_COMM_WORLD,&status);
  //       printf("received %d\n", m_size-1);
  //   }
  // }
  // else{
  //    MPI_Send(&C[jstart], jend-jstart, MPI_FLOAT, 0, 1000, MPI_COMM_WORLD);
  // }
  // MPI_Barrier(MPI_COMM_WORLD);
  free(transposeB);  
}
