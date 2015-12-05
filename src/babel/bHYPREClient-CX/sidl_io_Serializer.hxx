// 
// File:          sidl_io_Serializer.hxx
// Symbol:        sidl.io.Serializer-v0.9.15
// Symbol Type:   interface
// Babel Version: 1.0.4
// Release:       $Name: V2-4-0b $
// Revision:      @(#) $Id: sidl_io_Serializer.hxx,v 1.4 2007/09/27 19:55:46 painter Exp $
// Description:   Client-side glue code for sidl.io.Serializer
// 
// Copyright (c) 2000-2002, The Regents of the University of California.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the Components Team <components@llnl.gov>
// All rights reserved.
// 
// This file is part of Babel. For more information, see
// http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
// for Our Notice and the LICENSE file for the GNU Lesser General Public
// License.
// 
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License (as published by
// the Free Software Foundation) version 2.1 dated February 1999.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// conditions of the GNU Lesser General Public License for more details.
// 
// You should have recieved a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_sidl_io_Serializer_hxx
#define included_sidl_io_Serializer_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace sidl { 
  namespace io { 

    class Serializer;
  } // end namespace io
} // end namespace sidl

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::sidl::io::Serializer >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class RuntimeException;
} // end namespace sidl

namespace sidl { 
  namespace io { 

    class Serializable;
  } // end namespace io
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_sidl_io_Serializer_IOR_h
#include "sidl_io_Serializer_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
namespace sidl {
  namespace rmi {
    class Call;
    class Return;
    class Ticket;
  }
  namespace rmi {
    class InstanceHandle;
  }
}
namespace sidl { 
  namespace io { 

    /**
     * Symbol "sidl.io.Serializer" (version 0.9.15)
     * 
     * Standard interface for packing Babel types
     */
    class Serializer: public virtual ::sidl::BaseInterface {

      //////////////////////////////////////////////////
      // 
      // Special methods for throwing exceptions
      // 

    private:
      static 
      void
      throwException0(
        struct sidl_BaseInterface__object *_exception
      )
        // throws:
      ;

      //////////////////////////////////////////////////
      // 
      // User Defined Methods
      // 

    public:
      /**
       * user defined non-static method
       */
      void
      packBool (
        /* in */const ::std::string& key,
        /* in */bool value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packChar (
        /* in */const ::std::string& key,
        /* in */char value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packInt (
        /* in */const ::std::string& key,
        /* in */int32_t value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packLong (
        /* in */const ::std::string& key,
        /* in */int64_t value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packOpaque (
        /* in */const ::std::string& key,
        /* in */void* value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packFloat (
        /* in */const ::std::string& key,
        /* in */float value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packDouble (
        /* in */const ::std::string& key,
        /* in */double value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packFcomplex (
        /* in */const ::std::string& key,
        /* in */const ::std::complex<float>& value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packDcomplex (
        /* in */const ::std::string& key,
        /* in */const ::std::complex<double>& value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packString (
        /* in */const ::std::string& key,
        /* in */const ::std::string& value
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packSerializable (
        /* in */const ::std::string& key,
        /* in */::sidl::io::Serializable value
      )
      ;



      /**
       *  
       * pack arrays of values.  It is possible to ensure an array is
       * in a certain order by passing in ordering and dimension
       * requirements.  ordering should represent a value in the
       * sidl_array_ordering enumeration in sidlArray.h If either
       * argument is 0, it means there is no restriction on that
       * aspect.  The boolean reuse_array flag is set to true if the
       * remote unserializer should try to reuse the array that is
       * passed into it or not.
       */
      void
      packBoolArray (
        /* in */const ::std::string& key,
        /* in array<bool> */::sidl::array<bool> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packCharArray (
        /* in */const ::std::string& key,
        /* in array<char> */::sidl::array<char> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packIntArray (
        /* in */const ::std::string& key,
        /* in array<int> */::sidl::array<int32_t> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packLongArray (
        /* in */const ::std::string& key,
        /* in array<long> */::sidl::array<int64_t> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packOpaqueArray (
        /* in */const ::std::string& key,
        /* in array<opaque> */::sidl::array<void*> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packFloatArray (
        /* in */const ::std::string& key,
        /* in array<float> */::sidl::array<float> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packDoubleArray (
        /* in */const ::std::string& key,
        /* in array<double> */::sidl::array<double> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packFcomplexArray (
        /* in */const ::std::string& key,
        /* in array<fcomplex> */::sidl::array< ::sidl::fcomplex> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packDcomplexArray (
        /* in */const ::std::string& key,
        /* in array<dcomplex> */::sidl::array< ::sidl::dcomplex> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packStringArray (
        /* in */const ::std::string& key,
        /* in array<string> */::sidl::array< ::std::string> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packGenericArray (
        /* in */const ::std::string& key,
        /* in array<> */::sidl::basearray value,
        /* in */bool reuse_array
      )
      ;


      /**
       * user defined non-static method
       */
      void
      packSerializableArray (
        /* in */const ::std::string& key,
        /* in array<sidl.io.Serializable> */::sidl::array< 
          ::sidl::io::Serializable> value,
        /* in */int32_t ordering,
        /* in */int32_t dimen,
        /* in */bool reuse_array
      )
      ;



      //////////////////////////////////////////////////
      // 
      // End User Defined Methods
      // (everything else in this file is specific to
      //  Babel's C++ bindings)
      // 

    public:
      typedef struct sidl_io_Serializer__object ior_t;
      typedef struct sidl_io_Serializer__external ext_t;
      typedef struct sidl_io_Serializer__sepv sepv_t;

      // default constructor
      Serializer() { 
        sidl_io_Serializer_IORCache = NULL;
      }

      // RMI connect
      static inline ::sidl::io::Serializer _connect( /*in*/ const std::string& 
        url ) { 
        return _connect(url, true);
      }

      // RMI connect 2
      static ::sidl::io::Serializer _connect( /*in*/ const std::string& url, 
        /*in*/ const bool ar  );

      // default destructor
      virtual ~Serializer () { }

      // copy constructor
      Serializer ( const Serializer& original );

      // assignment operator
      Serializer& operator= ( const Serializer& rhs );

      // conversion from ior to C++ class
      Serializer ( Serializer::ior_t* ior );

      // Alternate constructor: does not call addRef()
      // (sets d_weak_reference=isWeak)
      // For internal use by Impls (fixes bug#275)
      Serializer ( Serializer::ior_t* ior, bool isWeak );

      inline ior_t* _get_ior() const throw() {
        if(!sidl_io_Serializer_IORCache) { 
          sidl_io_Serializer_IORCache = ::sidl::io::Serializer::_cast((
            void*)d_self);
          if (sidl_io_Serializer_IORCache) {
            struct sidl_BaseInterface__object *throwaway_exception;
            (sidl_io_Serializer_IORCache->d_epv->f_deleteRef)(
              sidl_io_Serializer_IORCache->d_object, &throwaway_exception);  
          }  
        }
        return sidl_io_Serializer_IORCache;
      }

      void _set_ior( ior_t* ptr ) throw () { 
        d_self = reinterpret_cast< void*>(ptr);
      }

      bool _is_nil() const throw () { return (d_self==0); }

      bool _not_nil() const throw () { return (d_self!=0); }

      bool operator !() const throw () { return (d_self==0); }

      static inline const char * type_name() throw () { return 
        "sidl.io.Serializer";}

      static struct sidl_io_Serializer__object* _cast(const void* src);

      // execute member function by name
      void _exec(const std::string& methodName,
                 ::sidl::rmi::Call& inArgs,
                 ::sidl::rmi::Return& outArgs);

      /**
       * Get the URL of the Implementation of this object (for RMI)
       */
      ::std::string
      _getURL() // throws:
      //     ::sidl::RuntimeException
      ;


      /**
       * Method to set whether or not method hooks should be invoked.
       */
      void
      _set_hooks (
        /* in */bool on
      )
      // throws:
      //     ::sidl::RuntimeException
      ;

      // return true iff object is remote
      bool _isRemote() const { 
        ior_t* self = const_cast<ior_t*>(_get_ior() );
        struct sidl_BaseInterface__object *throwaway_exception;
        return (*self->d_epv->f__isRemote)(self, &throwaway_exception) == TRUE;
      }

      // return true iff object is local
      bool _isLocal() const {
        return !_isRemote();
      }

    protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::sidl::NullIORException );


      //////////////////////////////////////////////////
      // 
      // Locally Cached IOR pointer
      // 

    protected:
      mutable ior_t* sidl_io_Serializer_IORCache;
    }; // end class Serializer
  } // end namespace io
} // end namespace sidl

extern "C" {


#pragma weak sidl_io_Serializer__connectI

#pragma weak sidl_io_Serializer__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct sidl_io_Serializer__object*
  sidl_io_Serializer__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct sidl_io_Serializer__object*
  sidl_io_Serializer__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::io::Serializer > {
    typedef array< ::sidl::io::Serializer > cxx_array_t;
    typedef ::sidl::io::Serializer cxx_item_t;
    typedef struct sidl_io_Serializer__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct sidl_io_Serializer__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::io::Serializer > > iterator;
    typedef const_array_iter< array_traits< ::sidl::io::Serializer > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::io::Serializer >: public interface_array< array_traits< 
    ::sidl::io::Serializer > > {
  public:
    typedef interface_array< array_traits< ::sidl::io::Serializer > > Base;
    typedef array_traits< ::sidl::io::Serializer >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::sidl::io::Serializer >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::sidl::io::Serializer >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::sidl::io::Serializer >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::io::Serializer >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_io_Serializer__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::io::Serializer >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::sidl::io::Serializer >&
    operator =( const array< ::sidl::io::Serializer >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#ifndef included_sidl_io_Serializable_hxx
#include "sidl_io_Serializable.hxx"
#endif
#endif
