#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
//#include <cuda_bf16.h>
#include <iostream>
#include <assert.h>

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline half2 pair2pack(half a, half b){
  return __halves2half2(a, b);
}

template<typename T>
__device__ inline T int2T(int a);

template<>
__device__ inline half int2T<half>(int a){
  return __int2half_rn(a);
}

__device__ inline half2 TtoT2(half a){
  return __half2half2(a);
}

template<typename T>
__device__ inline T float2T(float a);

template<>
__device__ inline half float2T<half>(float a){
  return __float2half(a);
}

__device__ inline float T2float(half a){
  return __half2float(a);
}

template<typename T>
__device__ inline T getzero();

template<>
__device__ inline half getzero<half>(){
  return __ushort_as_half((unsigned short)0x0000U);
}

template<typename T>
__device__ inline T getone();

template<>
__device__ inline half getone<half>(){
  return __ushort_as_half((unsigned short)0x3C00U);
}

template<typename T>
__device__ inline T hneg(T a){
  return __hneg(a);
}

template<typename T>
__device__ inline T hadd(T a, T b){
  return __hadd(a, b);
}

template<typename T>
__device__ inline T hsub(T a, T b){
  return __hsub(a, b);
}

template<typename T>
__device__ inline T hmul(T a, T b){
  return __hmul(a, b);
}

template<typename T>
__device__ inline T hdiv(T a, T b){
  return __hdiv(a, b);
}

template<typename T>
__device__ inline T hfma(T a, T b, T c){
  return __hfma(a, b, c);
}

template<typename T>
__device__ inline T hfma2(T a, T b, T c){
  return __hfma2(a, b, c);
}
