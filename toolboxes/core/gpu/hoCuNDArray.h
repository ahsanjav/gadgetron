/** \file hoNDArray.h
    \brief CPU-based N-dimensional array (data container) for cpu->gpu->cpu (hoCu) solvers.

    The existence of this class is mainly due to providing unique array type for the hoCu based math in
    hoCuNDArray_operators.h, hoCuNDArray_elemwise.h, and hoCuNDArray_blas.h.
    Unfortunately C++ does not let a derived class inherit its base class's constructors, which consequently need redefinition.
*/

#pragma once

#include "hoNDArray.h"

namespace Gadgetron{

  template<class T> class hoCuNDArray: public hoNDArray<T> 
  {    
  public:
    
    hoCuNDArray() : hoNDArray<T>() {}
    
    hoCuNDArray( std::vector<unsigned int> *dimensions ) : hoNDArray<T>(dimensions) {}

    hoCuNDArray( std::vector<unsigned int> *dimensions, T* data, bool delete_data_on_destruct = false ) :
      hoNDArray<T>(dimensions, data, delete_data_on_destruct){};
    
    hoCuNDArray( boost::shared_ptr< std::vector<unsigned int> > dimensions ) : hoNDArray<T>(dimensions) {}

    hoCuNDArray( boost::shared_ptr< std::vector<unsigned int> > dimensions, T* data, bool delete_data_on_destruct = false ) :
      hoNDArray<T>(dimensions, data, delete_data_on_destruct){};
    
    // Copy constructor
    hoCuNDArray(const hoNDArray<T>& a)
    {
      this->data_ = 0;
      this->dimensions_ = a.dimensions_;
      this->allocate_memory();
      memcpy( this->data_, a.data_, this->elements_*sizeof(T) );
    }    
  };
}
