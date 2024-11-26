#include <torch/all.h>

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  24;
const int BLOCKHEIGHT4B =  32;

const int MMBLOCKWIDTH  = 128;
const int MMBLOCKHEIGHT =  12;
const int MMBLOCKHEIGHT4B =  16;

const int MAXOUTLIER = 8;

int GetBLOCKWIDTH();


__global__ void VecQuant3MatMulKernel(
    const    float* __restrict__ vec,
    const      int* __restrict__ mat,
             float* __restrict__ mul,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
);

__global__ void MatQuant3DequantKernel(
    const      int* __restrict__ mat,
             float* __restrict__ out,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
);

__global__ void VecQuant3OutlierMatMulKernel(
    const    float* __restrict__ vec,
    const      int* __restrict__ mat,
             float* __restrict__ mul,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    const    float* __restrict__ outlierMat,
    const      int* __restrict__ outlieridx,
    const      int* __restrict__ outrow,
    const      int* __restrict__ cnt,
    int height,
    int width
);

__global__ void MatQuant3DequantOutlierKernel(
    const     int* __restrict__ mat,
               float* __restrict__ out,
    const      float* __restrict__ scales,
    const uint8_t* __restrict__ zeros,
    const      float* __restrict__ outlierMat,
    const     int* __restrict__ outlieridx,
    const     int* __restrict__ outrow,
    const     int* __restrict__ cnt,
    int height,
    int width
);

__global__ void VecQuant4MatMulKernel(
    const    float* __restrict__ vec,
    const      int* __restrict__ mat,
             float* __restrict__ mul,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
);


__global__ void MatQuant4DequantKernel(
    const      int* __restrict__ mat,
             float* __restrict__ out,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
);



__global__ void VecQuant4OutlierMatMulKernel(
    const    float* __restrict__ vec,
    const      int* __restrict__ mat,
             float* __restrict__ mul,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    const    float* __restrict__ outlierMat,
    const      int* __restrict__ outlieridx,
    const      int* __restrict__ outrow,
    const      int* __restrict__ cnt,
    int height,
    int width
);


void vecquant3matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
);

void vecquant3outliermatmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor outlierMat,
  torch::Tensor outlieridx,
  torch::Tensor outrow,
  torch::Tensor cnt
);


void matquant3dequant_cuda(
  torch::Tensor mat,
  torch::Tensor out,
  torch::Tensor scales,
  torch::Tensor zeros
);

void matquant3dequantoutlier_cuda( 
  torch::Tensor mat,
  torch::Tensor out,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor outlierMat,
  torch::Tensor outlieridx,
  torch::Tensor outrow,
  torch::Tensor cnt
);

void vecquant4matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
);

void vecquant4outliermatmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor outlierMat,
  torch::Tensor outlieridx,
  torch::Tensor outrow,
  torch::Tensor cnt
);

void matquant4dequant_cuda(
  torch::Tensor mat,
  torch::Tensor out,
  torch::Tensor scales,
  torch::Tensor zeros
);
