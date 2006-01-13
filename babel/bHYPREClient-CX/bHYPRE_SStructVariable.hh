// 
// File:          bHYPRE_SStructVariable.hh
// Symbol:        bHYPRE.SStructVariable-v1.0.0
// Symbol Type:   enumeration
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.SStructVariable
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_SStructVariable_hh
#define included_bHYPRE_SStructVariable_hh


#include "sidl_ucxx.hh"
#include "bHYPRE_SStructVariable_IOR.h"

namespace ucxx { 
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
} // end namespace ucxx


struct bHYPRE_SStructVariable__array;
namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::bHYPRE::SStructVariable > {
      typedef array< ::ucxx::bHYPRE::SStructVariable > cxx_array_t;
      typedef ::ucxx::bHYPRE::SStructVariable cxx_item_t;
      typedef struct bHYPRE_SStructVariable__array ior_array_t;
      typedef sidl_int__array ior_array_internal_t;
      typedef enum bHYPRE_SStructVariable__enum ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type& reference;
      typedef value_type* pointer;
      typedef const value_type& const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::bHYPRE::SStructVariable > > 
        iterator;
      typedef const_array_iter< array_traits< ::ucxx::bHYPRE::SStructVariable > 
        > const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::bHYPRE::SStructVariable >: public enum_array< 
      array_traits< ::ucxx::bHYPRE::SStructVariable > > {
    public:
      typedef enum_array< array_traits< ::ucxx::bHYPRE::SStructVariable > > 
        Base;
      typedef array_traits< ::ucxx::bHYPRE::SStructVariable >::cxx_array_t      
        cxx_array_t;
      typedef array_traits< ::ucxx::bHYPRE::SStructVariable >::cxx_item_t       
        cxx_item_t;
      typedef array_traits< ::ucxx::bHYPRE::SStructVariable >::ior_array_t      
        ior_array_t;
      typedef array_traits< ::ucxx::bHYPRE::SStructVariable 
        >::ior_array_internal_t ior_array_internal_t;
      typedef array_traits< ::ucxx::bHYPRE::SStructVariable >::ior_item_t       
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct bHYPRE_SStructVariable__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::bHYPRE::SStructVariable > &src) {
        d_array = src.d_array;
        if (d_array) addRef();
      }

      /**
       * assignment
       */
      array< ::ucxx::bHYPRE::SStructVariable >&
      operator =( const array< ::ucxx::bHYPRE::SStructVariable > &rhs) {
        if (d_array != rhs.d_array) {
          if (d_array) deleteRef();
          d_array = rhs.d_array;
          if (d_array) addRef();
        }
        return *this;
      }

    };
  }

  }


  #endif
