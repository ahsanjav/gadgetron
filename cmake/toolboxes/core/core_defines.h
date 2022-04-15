/** \file core_defines.h
    \brief Autogenerated header providing definitions of __host__, __device__, and __inline__ for systems on which Cuda is not installed.
*/

#pragma once

// Notice:
// -------
//
// The header cpucore_defines.h is autogenerated 
// by cmake from cpucore_defines.h.in
//

// Definition of Cuda availability passed to C++
//

#define GADGETRON_CUDA_IS_AVAILABLE 1

// Used Cuda host definitions if availble.
// Otherwise we leave them empty (as no device code is compiled anyway).

#ifdef __CUDACC__
#include "cuda_runtime.h"
//Here we bypass the CUDA 9 issues caused when CUDACC was deprecated
#ifdef __CUDACC_VER__
#ifdef __CUDACC_VER_MAJOR__
#undef __CUDACC_VER__
#define __CUDACC_VER__ (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100)
#endif
#endif
#else
#if !defined(__host__)
#define __host__
#endif
#if !defined(__device__)
#define __device__
#endif
#define __inline__ inline
#endif
