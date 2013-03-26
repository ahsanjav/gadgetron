/** \file cuNDArray_elemwise.h
    \brief Element-wise math operations on the cuNDArray class.
    
    cuNDArray_elementwise.h defines element-wise array operations on the cuNDArray class.
    Many of the provided functions come in two flavours:
    1) A function that returns a smart pointer to a new array holding the result of the element-wise operation, and
    2) A function that perform in-place element-wise computation replacing the input array.
    When both versions are available the in-place version is suffixed _inplace.
    Some functions (clear, fill, clamp, clamp_min, clamp_max, normalize, shrink1, shrinkd) are only provided as in-place operations,
    and they do not carry the _inplace suffix in order to keep user code compact.
    A few functions return a different type as its input array 
    (abs on complex data, real, imag, real_to_std_complex, real_to_complext) and consequently is not offered as an in place operation.
    The functions provided in cuNDArray_elemwise are deliberatly placed outside the NDArray derived classes
    - to allow the NDArray classes to be lightweight header only data containers for both the cpu and gpu instances
    - to allow for external library optimized implementations of the element-wise functions without adding such dependencies to the core data container
    The present cpu implementation is based on Thrust.
    The implementation is purposely split into a header and underlying implementation (.cpp) 
    as this allows specific instantiation of the supported template types.     
    The supported types are float, double Gadgetron::complext<float> and Gadgetron::complext<double> -- with some deliberate omissions.
    Arrays of type std::complex<float> and std::complex<double> are currently not supported since the thrust device functors cannot 
    link to std:: functions (as they are not declared as __device__). 
    However, arrays of type std::complex are binary compatible with arrays of type Gadgetron::complext (for which we have support)
    and can safely be cast to such.
*/

#pragma once

#include "cuNDArray.h"
#include "gpucore_export.h"

namespace Gadgetron{

  /**
   * @brief Calculates the element-wise absolute values (l2 norm) of the array entries
   * @param[in] x Input array.
   * @return A new array containing the element-wise absolute values of the input.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<typename realType<T>::Type> > abs( cuNDArray<T> *x );

  /**
   * @brief Calculates the element-wise absolute values (l2 norm) of the array entries (in place).
   * @param[in,out] x Input and output array.
   */
  template<class T> EXPORTGPUCORE void abs_inplace( cuNDArray<T> *x );
    
  /**
   * @brief Calculates the element-wise sqrt of the array entries.
   * @param[in] x Input array.
   * @return A new array containing the element-wise sqrt of the input.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<T> > sqrt( cuNDArray<T> *x );

  /**
   * @brief Calculates the element-wise sqrt of the array entries (in place).
   * @param[in,out] x Input and output array.
   */
  template<class T> EXPORTGPUCORE void sqrt_inplace( cuNDArray<T> *x );
  
  /**
   * @brief Calculates the element-wise square of the array entries.
   * @param[in] x Input array.
   * @return A new array containing the element-wise square of the input.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<T> > square( cuNDArray<T> *x );
    
  /**
   * @brief Calculates the element-wise square of the array entries (in place).
   * @param[in,out] x Input and output array.
   */
  template<class T> EXPORTGPUCORE void square_inplace( cuNDArray<T> *x );
    
  /**
   * @brief Calculates the element-wise reciprocal of the array entries.
   * @param[in] x Input array.
   * @return A new array containing the element-wise reciprocal of the input.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<T> > reciprocal( cuNDArray<T> *x );
  
  /**
   * @brief Calculates the element-wise reciprocal of the array entries (in place).
   * @param[in,out] x Input and output array.
   */
  template<class T> EXPORTGPUCORE void reciprocal_inplace( cuNDArray<T> *x );
  
  /**
   * @brief Calculates the element-wise reciprocal sqrt of the array entries.
   * @param[in] x Input array.
   * @return A new array containing the element-wise reciprocal sqrt of the input.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<T> > reciprocal_sqrt( cuNDArray<T> *x );
  
  /**
   * @brief Calculates the element-wise reciprocal sqrt of the array entries (in place).
   * @param[in,out] x Input and output array.
   */
  template<class T> EXPORTGPUCORE void reciprocal_sqrt_inplace( cuNDArray<T> *x );
  
  /**
   * @brief Calculates the elementwise signum function on the array.
   * @param[in] x Input array.
   * @return A new array containing the element-wise sgn of the input.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<T> > sgn( cuNDArray<T> *x );
  
  /**
   * @brief Calculates the elementwise signum function on the array (in place).
   * @param[in,out] x Input and output array.
   */
  template<class T> EXPORTGPUCORE void sgn_inplace( cuNDArray<T> *x );

  /**
   * @brief Extract the real component from a complex array.
   * @param[in] x Input array.
   * @return A new array of the real component of the complex array.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<T> > real( cuNDArray< std::complex<T> > *x );

  /**
   * @brief Extract the real component from a complex array.
   * @param[in] x Input array.
   * @return A new array of the real component of the complex array.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<T> > real( cuNDArray< complext<T> > *x );

  /**
   * @brief Extract the imaginary component from a complex array.
   * @param[in] x Input array.
   * @return A new array of the imaginary component of the complex array.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<T> > imag( cuNDArray< std::complex<T> > *x );

  /**
   * @brief Extract the imaginary component from a complex array.
   * @param[in] x Input array.
   * @return A new array of the imaginary component of the complex array.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray<T> > imag( cuNDArray< complext<T> > *x );

  /**
   * @brief Construct a complex array from a real array.
   * @param[in] x Input array.
   * @return A new complex array containing the input array in the real component and zeros in the imaginary component.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray< std::complex<T> > > real_to_std_complex( cuNDArray<T> *x );
  
  /**
   * @brief Construct a complex array from a real array.
   * @param[in] x Input array.
   * @return A new complex array containing the input array in the real component and zeros in the imaginary component.
   */
  template<class T> EXPORTGPUCORE boost::shared_ptr< cuNDArray< complext<T> > > real_to_complext( cuNDArray<T> *x );

  //
  // From hereon the functions are all in-place although without the _inplace suffix...
  //

  /**
   * @brief Clears the array to all zeros ( in place). Faster than fill.
   * @param[in,out] x Input and output array.
   */
  template<class T> EXPORTGPUCORE void clear( cuNDArray<T> *x );

  /**
   * @brief Fills the array with a user provided constant value (in place).
   * @param[in,out] x Input and output array.
   * @param[in] val Fill value.
   */
  template<class T> EXPORTGPUCORE void fill( cuNDArray<T> *x, T val );

  /**
   * @brief Clamps all values in the array to the minimum and maximum values specified (in place).
   * @param[in,out] x Input and output array.
   * @param[in] min minimum value.
   * @param[in] max maximum value.
   */
  template<class T> EXPORTGPUCORE void clamp( cuNDArray<T> *x, T min, T max );

  /**
   * @brief Clamps all values in the array to a minimum value allowed (in place).
   * @param[in,out] x Input and output array.
   * @param[in] min Minimum value.
   */
  template<class T> EXPORTGPUCORE void clamp_min( cuNDArray<T> *x, T min );

  /**
   * @brief Clamps all values in the array to a maximum value allowed (in place).
   * @param[in,out] x Input and output array.
   * @param[in] max Maximum value.
   */
  template<class T> EXPORTGPUCORE void clamp_max( cuNDArray<T> *x, T max );

  /**
   * @brief In place normalization (scaling) to a new maximum absolute array value val.
   * @param[in,out] x Input and output array.
   * @param[in] val New maximum absolute array value (according to the l2-norm)
   */  
  template<class T> EXPORTGPUCORE void normalize( cuNDArray<T> *x, typename realType<T>::Type val = typename realType<T>::Type(1) );

  /**
   * @brief In place shrinkage (soft thresholding), i.e. shrink(x,gamma) = x/abs(x)*max(abs(x)-gamma,0).
   * @param[in,out] x Input and output array.
   * @param[in] gamma Shrinkage control parameter
   */  
  template<class T> EXPORTGPUCORE void shrink1( cuNDArray<T> *x, typename realType<T>::Type gamma );

  /**
   * @brief In place shrinkage (soft thresholding, multi-dimensional), i.e. shrink(x,gamma,s) = x/s*max(s-gamma,0).
   * @param[in,out] x Input and output array.
   * @param[in] s Input array, normalization.
   * @param[in] gamma Shrinkage control parameter
   */  
  template<class T> EXPORTGPUCORE void shrinkd ( cuNDArray<T> *x, cuNDArray<typename realType<T>::Type> *s, typename realType<T>::Type gamma );
}
