// 
// File:          bHYPRE_SStructVariable.hxx
// Symbol:        bHYPRE.SStructVariable-v1.0.0
// Symbol Type:   enumeration
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.SStructVariable
// 
// WARNING: Automatically generated; changes will be lost
// 

#ifndef included_bHYPRE_SStructVariable_hxx
#define included_bHYPRE_SStructVariable_hxx


#include "sidl_cxx.hxx"
#include "bHYPRE_SStructVariable_IOR.h"

namespace bHYPRE { 

  enum SStructVariable {
    SStructVariable_UNDEFINED = -1,

    SStructVariable_CELL = 0,

    SStructVariable_NODE = 1,

    SStructVariable_XFACE = 2,

    SStructVariable_YFACE = 3,

    SStructVariable_ZFACE = 4,

    SStructVariable_XEDGE = 5,

    SStructVariable_YEDGE = 6,

    SStructVariable_ZEDGE = 7

  };

} // end namespace bHYPRE


struct bHYPRE_SStructVariable__array;
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::SStructVariable > {
    typedef array< ::bHYPRE::SStructVariable > cxx_array_t;
    typedef ::bHYPRE::SStructVariable cxx_item_t;
    typedef struct bHYPRE_SStructVariable__array ior_array_t;
    typedef sidl_int__array ior_array_internal_t;
    typedef enum bHYPRE_SStructVariable__enum ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef const value_type& const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::SStructVariable > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::SStructVariable > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::SStructVariable >: public enum_array< array_traits< 
    ::bHYPRE::SStructVariable > > {
  public:
    typedef enum_array< array_traits< ::bHYPRE::SStructVariable > > Base;
    typedef array_traits< ::bHYPRE::SStructVariable >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::SStructVariable >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::SStructVariable >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::SStructVariable >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::SStructVariable >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_SStructVariable__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::SStructVariable > &src) {
      d_array = src.d_array;
      if (d_array) addRef();
    }

    /**
     * assignment
     */
    array< ::bHYPRE::SStructVariable >&
    operator =( const array< ::bHYPRE::SStructVariable > &rhs) {
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
