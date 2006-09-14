// 
// File:          bHYPRE_ErrorCode.hxx
// Symbol:        bHYPRE.ErrorCode-v1.0.0
// Symbol Type:   enumeration
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.ErrorCode
// 
// WARNING: Automatically generated; changes will be lost
// 

#ifndef included_bHYPRE_ErrorCode_hxx
#define included_bHYPRE_ErrorCode_hxx


#include "sidl_cxx.hxx"
#include "bHYPRE_ErrorCode_IOR.h"

namespace bHYPRE { 

  enum ErrorCode {
    ErrorCode_HYPRE_ERROR_GENERIC = 1,

    ErrorCode_HYPRE_ERROR_MEMORY = 2,

    ErrorCode_HYPRE_ERROR_ARG = 4,

    ErrorCode_HYPRE_ERROR_CONV = 256

  };

} // end namespace bHYPRE


struct bHYPRE_ErrorCode__array;
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::ErrorCode > {
    typedef array< ::bHYPRE::ErrorCode > cxx_array_t;
    typedef ::bHYPRE::ErrorCode cxx_item_t;
    typedef struct bHYPRE_ErrorCode__array ior_array_t;
    typedef sidl_int__array ior_array_internal_t;
    typedef enum bHYPRE_ErrorCode__enum ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef const value_type& const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::ErrorCode > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::ErrorCode > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::ErrorCode >: public enum_array< array_traits< 
    ::bHYPRE::ErrorCode > > {
  public:
    typedef enum_array< array_traits< ::bHYPRE::ErrorCode > > Base;
    typedef array_traits< ::bHYPRE::ErrorCode >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::ErrorCode >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::ErrorCode >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::ErrorCode >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::ErrorCode >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_ErrorCode__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::ErrorCode > &src) {
      d_array = src.d_array;
      if (d_array) addRef();
    }

    /**
     * assignment
     */
    array< ::bHYPRE::ErrorCode >&
    operator =( const array< ::bHYPRE::ErrorCode > &rhs) {
      if (d_array != rhs.d_array) {
        if (d_array) deleteRef();
        d_array = rhs.d_array;
        if (d_array) addRef();
      }
      return *this;
    }

  };
}


#endif
