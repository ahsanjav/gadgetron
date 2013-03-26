#include "hoNDArray_blas.h"
#include "complext.h"
#include <complex>

using namespace Gadgetron;

template<class T> T 
Gadgetron::dot( hoNDArray<T> *x, hoNDArray<T> *y, bool cc )
{
  arma::Col<typename stdType<T>::Type> xM = as_arma_col(x);
  arma::Col<typename stdType<T>::Type> yM = as_arma_col(y);
  typename stdType<T>::Type res = (cc) ? arma::cdot(xM,yM) : arma::dot(xM,yM);
  return *((T*)(&res));
}

template<class T> typename realType<T>::Type 
Gadgetron::asum( hoNDArray<T> *x )
{
  typedef typename realType<T>::Type realT;
  arma::Col<typename stdType<T>::Type> xM = as_arma_col(x);
  return realT(arma::norm(xM,1));
}

template<class T> T 
Gadgetron::asum( hoNDArray< std::complex<T> > *x )
{
  return arma::norm(arma::abs(real(as_arma_col(x)))+arma::abs(imag(as_arma_col(x))),1);
}

template<class T> T 
Gadgetron::asum( hoNDArray< complext<T> > *x )
{
  return arma::norm(arma::abs(real(as_arma_col(x)))+arma::abs(imag(as_arma_col(x))),1);
}

template<class T> typename realType<T>::Type 
Gadgetron::nrm2( hoNDArray<T> *x )
{
  typedef typename realType<T>::Type realT;
  arma::Col<typename stdType<T>::Type> xM = as_arma_col(x);
  return realT(arma::norm(xM,2));
}
  
template<class T> unsigned int 
Gadgetron::amin( hoNDArray<T> *x )
{
  typedef typename realType<T>::Type realT;
  arma::Col<realT> xM = arma::abs(as_arma_col(x));
  unsigned int idx;
  realT min = xM.min(idx);
  return idx;
}

template<class T> unsigned int 
Gadgetron::amin( hoNDArray< std::complex<T> > *x )
{
  arma::Col<T> xM = arma::abs(real(as_arma_col(x)))+arma::abs(imag(as_arma_col(x)));
  unsigned int idx;
  T min = xM.min(idx);
  return idx;
}

template<class T> unsigned int 
Gadgetron::amin( hoNDArray< complext<T> > *x )
{
  arma::Col<T> xM = arma::abs(real(as_arma_col(x)))+arma::abs(imag(as_arma_col(x)));
  unsigned int idx;
  T min = xM.min(idx);
  return idx;
}

template<class T> unsigned int 
Gadgetron::amax( hoNDArray<T> *x )
{
  typedef typename realType<T>::Type realT;
  arma::Col<realT> xM = arma::abs(as_arma_col(x));
  unsigned int idx;
  realT max = xM.max(idx);
  return idx;
}

template<class T> unsigned int 
Gadgetron::amax( hoNDArray< std::complex<T> > *x )
{
  arma::Col<T> xM = arma::abs(real(as_arma_col(x)))+arma::abs(imag(as_arma_col(x)));
  unsigned int idx;
  T max = xM.max(idx);
  return idx;
}

template<class T> unsigned int 
Gadgetron::amax( hoNDArray< complext<T> > *x )
{
  arma::Col<T> xM = arma::abs(real(as_arma_col(x)))+arma::abs(imag(as_arma_col(x)));
  unsigned int idx;
  T max = xM.max(idx);
  return idx;
}

template<class T> void 
Gadgetron::axpy( T a, hoNDArray<T> *x, hoNDArray<T> *y )
{
  typedef typename stdType<T>::Type stdT;
  arma::Col<stdT> xM = as_arma_col(x);
  arma::Col<stdT> yM = as_arma_col(y);
  stdT a2 = *((stdT*)(&a));
  yM += (a2*xM);
}

//
// Instantiation
//

template EXPORTCPUCOREMATH float dot<float>( hoNDArray<float>*, hoNDArray<float>*, bool );
template EXPORTCPUCOREMATH float asum<float>( hoNDArray<float>* );
template EXPORTCPUCOREMATH float nrm2<float>( hoNDArray<float>* );
template EXPORTCPUCOREMATH unsigned int amin<float>( hoNDArray<float>* );
template EXPORTCPUCOREMATH unsigned int amax<float>( hoNDArray<float>* );
template EXPORTCPUCOREMATH void axpy<float>( float, hoNDArray<float>*, hoNDArray<float>* );

template EXPORTCPUCOREMATH double dot<double>( hoNDArray<double>*, hoNDArray<double>*, bool );
template EXPORTCPUCOREMATH double asum<double>( hoNDArray<double>* );
template EXPORTCPUCOREMATH double nrm2<double>( hoNDArray<double>* );
template EXPORTCPUCOREMATH unsigned int amin<double>( hoNDArray<double>* );
template EXPORTCPUCOREMATH unsigned int amax<double>( hoNDArray<double>* );
template EXPORTCPUCOREMATH void axpy<double>( double, hoNDArray<double>*, hoNDArray<double>* );

template EXPORTCPUCOREMATH std::complex<float> dot< std::complex<float> >( hoNDArray< std::complex<float> >*, hoNDArray< std::complex<float> >*, bool );
template EXPORTCPUCOREMATH float asum<float>( hoNDArray< std::complex<float> >* );
template EXPORTCPUCOREMATH float nrm2< std::complex<float> >( hoNDArray< std::complex<float> >* );
template EXPORTCPUCOREMATH unsigned int amin<float>( hoNDArray< std::complex<float> >* );
template EXPORTCPUCOREMATH unsigned int amax<float>( hoNDArray< std::complex<float> >* );
template EXPORTCPUCOREMATH void axpy< std::complex<float> >( std::complex<float> , hoNDArray< std::complex<float> >*, hoNDArray< std::complex<float> >* );

template EXPORTCPUCOREMATH std::complex<double> dot< std::complex<double> >( hoNDArray< std::complex<double> >*, hoNDArray< std::complex<double> >*, bool );
template EXPORTCPUCOREMATH double asum<double>( hoNDArray< std::complex<double> >* );
template EXPORTCPUCOREMATH double nrm2< std::complex<double> >( hoNDArray< std::complex<double> >* );
template EXPORTCPUCOREMATH unsigned int amin<double>( hoNDArray< std::complex<double> >* );
template EXPORTCPUCOREMATH unsigned int amax<double>( hoNDArray< std::complex<double> >* );
template EXPORTCPUCOREMATH void axpy< std::complex<double> >( std::complex<double> , hoNDArray< std::complex<double> >*, hoNDArray< std::complex<double> >* );

template EXPORTCPUCOREMATH complext<float> dot< complext<float> >( hoNDArray< complext<float> >*, hoNDArray< complext<float> >*, bool );
template EXPORTCPUCOREMATH float asum<float>( hoNDArray< complext<float> >* );
template EXPORTCPUCOREMATH float nrm2< complext<float> >( hoNDArray< complext<float> >* );
template EXPORTCPUCOREMATH unsigned int amin<float>( hoNDArray< complext<float> >* );
template EXPORTCPUCOREMATH unsigned int amax<float>( hoNDArray< complext<float> >* );
template EXPORTCPUCOREMATH void axpy< complext<float> >( complext<float> , hoNDArray< complext<float> >*, hoNDArray< complext<float> >* );

template EXPORTCPUCOREMATH complext<double> dot< complext<double> >( hoNDArray< complext<double> >*, hoNDArray< complext<double> >*, bool );
template EXPORTCPUCOREMATH double asum<double>( hoNDArray< complext<double> >* );
template EXPORTCPUCOREMATH double nrm2< complext<double> >( hoNDArray< complext<double> >* );
template EXPORTCPUCOREMATH unsigned int amin<double>( hoNDArray< complext<double> >* );
template EXPORTCPUCOREMATH unsigned int amax<double>( hoNDArray< complext<double> >* );
template EXPORTCPUCOREMATH void axpy< complext<double> >( complext<double> , hoNDArray< complext<double> >*, hoNDArray< complext<double> >* );
