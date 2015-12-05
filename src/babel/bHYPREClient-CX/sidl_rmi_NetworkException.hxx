// 
// File:          sidl_rmi_NetworkException.hxx
// Symbol:        sidl.rmi.NetworkException-v0.9.15
// Symbol Type:   class
// Babel Version: 1.0.0
// Release:       $Name: V1-14-0b $
// Revision:      @(#) $Id: sidl_rmi_NetworkException.hxx,v 1.2 2006/09/14 21:52:16 painter Exp $
// Description:   Client-side glue code for sidl.rmi.NetworkException
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

#ifndef included_sidl_rmi_NetworkException_hxx
#define included_sidl_rmi_NetworkException_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace sidl { 
  namespace rmi { 

    class NetworkException;
  } // end namespace rmi
} // end namespace sidl

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::sidl::rmi::NetworkException >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class RuntimeException;
} // end namespace sidl

namespace sidl { 
  namespace io { 

    class Deserializer;
  } // end namespace io
} // end namespace sidl

namespace sidl { 
  namespace io { 

    class Serializer;
  } // end namespace io
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_sidl_rmi_NetworkException_IOR_h
#include "sidl_rmi_NetworkException_IOR.h"
#endif
#ifndef included_sidl_io_IOException_hxx
#include "sidl_io_IOException.hxx"
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
  namespace rmi { 

    /**
     * Symbol "sidl.rmi.NetworkException" (version 0.9.15)
     * 
     * Generic Network Exception
     */
    class NetworkException: public virtual ::sidl::io::IOException {

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
      int32_t
      getHopCount() ;

      /**
       * user defined non-static method
       */
      void
      packObj (
        /* in */::sidl::io::Serializer ser
      )
      ;


      /**
       * user defined non-static method
       */
      void
      unpackObj (
        /* in */::sidl::io::Deserializer des
      )
      ;



      //////////////////////////////////////////////////
      // 
      // End User Defined Methods
      // (everything else in this file is specific to
      //  Babel's C++ bindings)
      // 

    public:
      typedef struct sidl_rmi_NetworkException__object ior_t;
      typedef struct sidl_rmi_NetworkException__external ext_t;
      typedef struct sidl_rmi_NetworkException__sepv sepv_t;

      // default constructor
      NetworkException() { }

      // static constructor
      static ::sidl::rmi::NetworkException _create();

      // RMI constructor
      static ::sidl::rmi::NetworkException _create( /*in*/ const std::string& 
        url );

      // RMI connect
      static inline ::sidl::rmi::NetworkException _connect( /*in*/ const 
        std::string& url ) { 
        return _connect(url, true);
      }

      // RMI connect 2
      static ::sidl::rmi::NetworkException _connect( /*in*/ const std::string& 
        url, /*in*/ const bool ar  );

      // default destructor
      virtual ~NetworkException () { }

      // copy constructor
      NetworkException ( const NetworkException& original );

      // assignment operator
      NetworkException& operator= ( const NetworkException& rhs );

      // conversion from ior to C++ class
      NetworkException ( NetworkException::ior_t* ior );

      // Alternate constructor: does not call addRef()
      // (sets d_weak_reference=isWeak)
      // For internal use by Impls (fixes bug#275)
      NetworkException ( NetworkException::ior_t* ior, bool isWeak );

      ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

      const ior_t* _get_ior() const throw () { return reinterpret_cast< 
        ior_t*>(d_self); }

      void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
        void*>(ptr); }

      bool _is_nil() const throw () { return (d_self==0); }

      bool _not_nil() const throw () { return (d_self!=0); }

      bool operator !() const throw () { return (d_self==0); }

      static inline const char * type_name() throw () { return 
        "sidl.rmi.NetworkException";}

      static struct sidl_rmi_NetworkException__object* _cast(const void* src);

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

    }; // end class NetworkException
  } // end namespace rmi
} // end namespace sidl

extern "C" {


  #pragma weak sidl_rmi_NetworkException__connectI

  #pragma weak sidl_rmi_NetworkException__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct sidl_rmi_NetworkException__object*
  sidl_rmi_NetworkException__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct sidl_rmi_NetworkException__object*
  sidl_rmi_NetworkException__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::rmi::NetworkException > {
    typedef array< ::sidl::rmi::NetworkException > cxx_array_t;
    typedef ::sidl::rmi::NetworkException cxx_item_t;
    typedef struct sidl_rmi_NetworkException__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct sidl_rmi_NetworkException__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::rmi::NetworkException > > 
      iterator;
    typedef const_array_iter< array_traits< ::sidl::rmi::NetworkException > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::rmi::NetworkException >: public interface_array< 
    array_traits< ::sidl::rmi::NetworkException > > {
  public:
    typedef interface_array< array_traits< ::sidl::rmi::NetworkException > > 
      Base;
    typedef array_traits< ::sidl::rmi::NetworkException >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::sidl::rmi::NetworkException >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::sidl::rmi::NetworkException >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::sidl::rmi::NetworkException >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::rmi::NetworkException >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_rmi_NetworkException__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::rmi::NetworkException >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::sidl::rmi::NetworkException >&
    operator =( const array< ::sidl::rmi::NetworkException >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#ifndef included_sidl_io_Deserializer_hxx
#include "sidl_io_Deserializer.hxx"
#endif
#ifndef included_sidl_io_Serializer_hxx
#include "sidl_io_Serializer.hxx"
#endif
#endif
