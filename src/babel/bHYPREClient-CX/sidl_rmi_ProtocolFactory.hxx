// 
// File:          sidl_rmi_ProtocolFactory.hxx
// Symbol:        sidl.rmi.ProtocolFactory-v0.9.15
// Symbol Type:   class
// Babel Version: 1.0.0
// Release:       $Name: V1-14-0b $
// Revision:      @(#) $Id: sidl_rmi_ProtocolFactory.hxx,v 1.2 2006/09/14 21:52:16 painter Exp $
// Description:   Client-side glue code for sidl.rmi.ProtocolFactory
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

#ifndef included_sidl_rmi_ProtocolFactory_hxx
#define included_sidl_rmi_ProtocolFactory_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace sidl { 
  namespace rmi { 

    class ProtocolFactory;
  } // end namespace rmi
} // end namespace sidl

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::sidl::rmi::ProtocolFactory >;
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

namespace sidl { 
  namespace rmi { 

    class InstanceHandle;
  } // end namespace rmi
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_IOR_h
#include "sidl_rmi_ProtocolFactory_IOR.h"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
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
     * Symbol "sidl.rmi.ProtocolFactory" (version 0.9.15)
     * 
     *  
     * This singleton class keeps a table of string prefixes
     * (e.g. "babel" or "proteus") to protocol implementations.  The
     * intent is to parse a URL (e.g. "babel://server:port/class") and
     * create classes that implement
     * <code>sidl.rmi.InstanceHandle</code>.
     */
    class ProtocolFactory: public virtual ::sidl::BaseClass {

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
       *  
       * Associate a particular prefix in the URL to a typeName
       * <code>sidl.Loader</code> can find.  The actual type is
       * expected to implement <code>sidl.rmi.InstanceHandle</code>
       * Return true iff the addition is successful.  (no collisions
       * allowed)
       */
      static bool
      addProtocol (
        /* in */const ::std::string& prefix,
        /* in */const ::std::string& typeName
      )
      ;



      /**
       * Return the typeName associated with a particular prefix.
       * Return empty string if the prefix
       */
      static ::std::string
      getProtocol (
        /* in */const ::std::string& prefix
      )
      ;



      /**
       * Remove a protocol from the active list.
       */
      static bool
      deleteProtocol (
        /* in */const ::std::string& prefix
      )
      ;



      /**
       * Create a new remote object and return an instance handle for that
       * object. 
       * The server and port number are in the url.  Return nil 
       * if protocol unknown or InstanceHandle.init() failed.
       */
      static ::sidl::rmi::InstanceHandle
      createInstance (
        /* in */const ::std::string& url,
        /* in */const ::std::string& typeName
      )
      ;



      /**
       *  
       * Create an new connection linked to an already existing
       * object on a remote server.  The server and port number are in
       * the url, the objectID is the unique ID of the remote object
       * in the remote instance registry.  Return null if protocol
       * unknown or InstanceHandle.init() failed.  The boolean addRef
       * should be true if connect should remotely addRef
       */
      static ::sidl::rmi::InstanceHandle
      connectInstance (
        /* in */const ::std::string& url,
        /* in */bool ar
      )
      ;



      /**
       *  
       * Request that a remote object be serialized to you.  The server 
       * and port number are in the url, the objectID is the unique ID 
       * of the remote object in the remote instance registry.  Return 
       * null if protocol unknown or InstanceHandle.init() failed.  
       */
      static ::sidl::io::Serializable
      unserializeInstance (
        /* in */const ::std::string& url
      )
      ;



      //////////////////////////////////////////////////
      // 
      // End User Defined Methods
      // (everything else in this file is specific to
      //  Babel's C++ bindings)
      // 

    public:
      typedef struct sidl_rmi_ProtocolFactory__object ior_t;
      typedef struct sidl_rmi_ProtocolFactory__external ext_t;
      typedef struct sidl_rmi_ProtocolFactory__sepv sepv_t;

      // default constructor
      ProtocolFactory() { }

      // static constructor
      static ::sidl::rmi::ProtocolFactory _create();

      // RMI constructor
      static ::sidl::rmi::ProtocolFactory _create( /*in*/ const std::string& 
        url );

      // RMI connect
      static inline ::sidl::rmi::ProtocolFactory _connect( /*in*/ const 
        std::string& url ) { 
        return _connect(url, true);
      }

      // RMI connect 2
      static ::sidl::rmi::ProtocolFactory _connect( /*in*/ const std::string& 
        url, /*in*/ const bool ar  );

      // default destructor
      virtual ~ProtocolFactory () { }

      // copy constructor
      ProtocolFactory ( const ProtocolFactory& original );

      // assignment operator
      ProtocolFactory& operator= ( const ProtocolFactory& rhs );

      // conversion from ior to C++ class
      ProtocolFactory ( ProtocolFactory::ior_t* ior );

      // Alternate constructor: does not call addRef()
      // (sets d_weak_reference=isWeak)
      // For internal use by Impls (fixes bug#275)
      ProtocolFactory ( ProtocolFactory::ior_t* ior, bool isWeak );

      ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

      const ior_t* _get_ior() const throw () { return reinterpret_cast< 
        ior_t*>(d_self); }

      void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
        void*>(ptr); }

      bool _is_nil() const throw () { return (d_self==0); }

      bool _not_nil() const throw () { return (d_self!=0); }

      bool operator !() const throw () { return (d_self==0); }

      static inline const char * type_name() throw () { return 
        "sidl.rmi.ProtocolFactory";}

      static struct sidl_rmi_ProtocolFactory__object* _cast(const void* src);

      // execute member function by name
      void _exec(const std::string& methodName,
                 ::sidl::rmi::Call& inArgs,
                 ::sidl::rmi::Return& outArgs);
      // exec static member function by name
      static void _sexec(const std::string& methodName,
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


      /**
       * Static Method to set whether or not method hooks should be invoked.
       */
      static void
      _set_hooks_static (
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

      static const sepv_t * _get_sepv() {
        return (*(_get_ext()->getStaticEPV))();
      }

    }; // end class ProtocolFactory
  } // end namespace rmi
} // end namespace sidl

extern "C" {


  #pragma weak sidl_rmi_ProtocolFactory__connectI

  #pragma weak sidl_rmi_ProtocolFactory__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct sidl_rmi_ProtocolFactory__object*
  sidl_rmi_ProtocolFactory__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct sidl_rmi_ProtocolFactory__object*
  sidl_rmi_ProtocolFactory__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::rmi::ProtocolFactory > {
    typedef array< ::sidl::rmi::ProtocolFactory > cxx_array_t;
    typedef ::sidl::rmi::ProtocolFactory cxx_item_t;
    typedef struct sidl_rmi_ProtocolFactory__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct sidl_rmi_ProtocolFactory__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::rmi::ProtocolFactory > > iterator;
    typedef const_array_iter< array_traits< ::sidl::rmi::ProtocolFactory > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::rmi::ProtocolFactory >: public interface_array< 
    array_traits< ::sidl::rmi::ProtocolFactory > > {
  public:
    typedef interface_array< array_traits< ::sidl::rmi::ProtocolFactory > > 
      Base;
    typedef array_traits< ::sidl::rmi::ProtocolFactory >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::sidl::rmi::ProtocolFactory >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::sidl::rmi::ProtocolFactory >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::sidl::rmi::ProtocolFactory >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::rmi::ProtocolFactory >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_rmi_ProtocolFactory__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::rmi::ProtocolFactory >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::sidl::rmi::ProtocolFactory >&
    operator =( const array< ::sidl::rmi::ProtocolFactory >&rhs ) { 
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
#ifndef included_sidl_rmi_InstanceHandle_hxx
#include "sidl_rmi_InstanceHandle.hxx"
#endif
#endif
