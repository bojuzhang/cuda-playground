#include <cassert>
#include <chrono>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>

// You may increase this value to test larger matrices
// But it will be slow on CPU
constexpr int MAXN = 4096;
constexpr int BLOCK_SIZE = 32;

/**
 * @brief A naive implementation of matrix multiplication on CPU.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
void naiveSgemm(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += a[m * K + k] * b[k * N + n];
      }
      c[m * N + n] = sum;
    }
  }
}

/**
 * @brief A naive implementation of matrix multiplication on GPU.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
__global__ void naiveSgemm2D(float *a, float *b, float *c, const int M,
                             const int N, const int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (m < M && n < N) {
    float sum = 0.0;
    for (int k = 0; k < K; ++k) {
      sum += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = sum;
  }
}

__global__ void optimizedSgemmKernel(float *a, float *b, float *c, 
                                      int M, int N, int K) {
    // more continuous
    const int block_x = blockIdx.y;
    const int block_y = blockIdx.x;
    const int thread_x = threadIdx.y;
    const int thread_y = threadIdx.x;
    
    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];
    
    const int x = block_x * BLOCK_SIZE + thread_x;
    const int y = block_y * BLOCK_SIZE + thread_y;
    
    float res = 0.0;
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE) {
      int a_x = x;
      int a_y = tile_idx + thread_y;
      if (a_x < M && a_y < K) {
        as[thread_x][thread_y] = a[a_x * K + a_y];
      } else {
        as[thread_x][thread_y] = 0.0;
      }
      
      int b_x = tile_idx + thread_x;
      int b_y = y;
      if (b_x < K && b_y < N) {
        bs[thread_x][thread_y] = b[b_x * N + b_y];
      } else {
        bs[thread_x][thread_y] = 0.0;
      }
      __syncthreads();

      for (int k = 0; k < BLOCK_SIZE; k++) {
        res += as[thread_x][k] * bs[k][thread_y];
      }
      __syncthreads();
    }
    if (x < M && y < N) {
      c[x * N + y] = res;
    }
}

/**
 * @brief Launch naiveSgemm2D kernel.
 */
void launchSgemm2D(float *a, float *b, float *c, const int M, const int N,
                   const int K) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  optimizedSgemmKernel<<<grid, block>>>(a, b, c, M, N, K);
}

void initialize(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  auto gen = std::mt19937(2024);
  auto dis = std::uniform_real_distribution<float>(-1.0, 1.0);
  for (int i = 0; i < M * K; ++i) {
    a[i] = dis(gen);
  }
  for (int i = 0; i < K * N; ++i) {
    b[i] = dis(gen);
  }
  for (int i = 0; i < M * N; ++i) {
    c[i] = 0.0;
  }
}

/** 
 * @brief Launch sgemm using cuBLAS
 */
void launchCublasSgemm(float *a, float *b, float *c, const int M, const int N,
                       const int K) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K,
              &beta, c, N);
}

bool compare(float *a, float *b, const int N, const int M) {
  for (int i = 0; i < N; ++i) {
    if (std::abs(a[i] - b[i]) > 1e-3) {
      printf("Mismatch at index %d %d: %f vs %f\n", i / M, i % M, a[i], b[i]);
      return false;
    }
  }
  printf("Results match\n");
  return true;
}


int main() {
  float *a, *b, *c;
  a = new float[MAXN * MAXN];
  b = new float[MAXN * MAXN];
  c = new float[MAXN * MAXN];
  initialize(a, b, c, MAXN, MAXN, MAXN);

  // ********** CPU **********
  // auto start = std::chrono::high_resolution_clock::now();
  // naiveSgemm(a, b, c, MAXN, MAXN, MAXN);
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed = end - start;
  // printf("CPU time: %.3fs\n", elapsed.count());

  float *d_a, *d_b, *d_c, *dd_c;
  cudaMalloc(&d_a, MAXN * MAXN * sizeof(float));
  cudaMalloc(&d_b, MAXN * MAXN * sizeof(float));
  cudaMalloc(&d_c, MAXN * MAXN * sizeof(float));
  cudaMalloc(&dd_c, MAXN * MAXN * sizeof(float));
  cudaMemcpy(d_a, a, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dd_c, c, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);

  // ********** GPU **********
  auto start = std::chrono::high_resolution_clock::now();
  launchSgemm2D(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("GPU time: %.3fs\n", elapsed.count());

  // ********** cuBLAS **********
  start = std::chrono::high_resolution_clock::now();
  launchCublasSgemm(d_a, d_b, dd_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  printf("cuBLAS time: %.3fs\n", elapsed.count());

  float *cc, *ccc;
  cc = new float[MAXN * MAXN];
  ccc = new float[MAXN * MAXN];
  cudaMemcpy(cc, d_c, MAXN * MAXN * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(ccc, dd_c, MAXN * MAXN * sizeof(float), cudaMemcpyDeviceToHost);

  compare(cc, ccc, MAXN * MAXN, MAXN);
}
