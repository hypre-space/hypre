// 
// File:          sidl_rmi_ProtocolFactory.hh
// Symbol:        sidl.rmi.ProtocolFactory-v0.9.3
// Symbol Type:   class
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
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
// babel-version = 0.10.12
// xml-url       = /home/painter/babel/share/babel-0.10.12/repository/sidl.rmi.ProtocolFactory-v0.9.3.xml
// 

#ifndef included_sidl_rmi_ProtocolFactory_hh
#define included_sidl_rmi_ProtocolFactory_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace sidl { 
    namespace rmi { 

      class ProtocolFactory;
    } // end namespace rmi
  } // end namespace sidl
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::sidl::rmi::ProtocolFactory >;
  }
} //closes ucxx Namespace
// 
// Forward declarations for method dependencies.
// 
namespace ucxx { 
  namespace sidl { 

    class BaseInterface;
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx { 
  namespace sidl { 

    class ClassInfo;
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx { 
  namespace sidl { 
    namespace rmi { 

      class InstanceHandle;
    } // end namespace rmi
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx { 
  namespace sidl { 
    namespace rmi { 

      class NetworkException;
    } // end namespace rmi
  } // end namespace sidl
} // end namespace ucxx

#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_IOR_h
#include "sidl_rmi_ProtocolFactory_IOR.h"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif
#ifndef included_sidl_rmi_NetworkException_hh
#include "sidl_rmi_NetworkException.hh"
#endif

namespace ucxx { 
  namespace sidl { 
    namespace rmi { 

      /**
       * Symbol "sidl.rmi.ProtocolFactory" (version 0.9.3)
       * 
       * This singleton class keeps a table of string prefixes (e.g. "babel" or "proteus")
       * to protocol implementations.  The intent is to parse a URL (e.g. "babel://server:port/class")
       * and create classes that implement <code>sidl.rmi.InstanceHandle</code>.
       */
      class ProtocolFactory: public virtual ::ucxx::sidl::BaseClass {

      //////////////////////////////////////////////////
      // 
      // User Defined Methods
      // 

    public:

      /**
       * Associate a particular prefix in the URL to a typeName <code>sidl.Loader</code> can find.
       * The actual type is expected to implement <code>sidl.rmi.InstanceHandle</code>
       * Return true iff the addition is successful.  (no collisions allowed)
       */
      static bool
      addProtocol (
        /* in */const ::std::string& prefix,
        /* in */const ::std::string& typeName
      )
      throw ( 
        ::ucxx::sidl::rmi::NetworkException
      );



      /**
       * Return the typeName associated with a particular prefix.
       * Return empty string if the prefix
       */
      static ::std::string
      getProtocol (
        /* in */const ::std::string& prefix
      )
      throw ( 
        ::ucxx::sidl::rmi::NetworkException
      );



      /**
       * Remove a protocol from the active list.
       */
      static bool
      deleteProtocol (
        /* in */const ::std::string& prefix
      )
      throw ( 
        ::ucxx::sidl::rmi::NetworkException
      );



      /**
       * Create a new remote object and a return an instance handle for that object. 
       * The server and port number are in the url.  Return nil 
       * if protocol unknown or InstanceHandle.init() failed.
       */
      static ::ucxx::sidl::rmi::InstanceHandle
      createInstance (
        /* in */const ::std::string& url,
        /* in */const ::std::string& typeName
      )
      throw ( 
        ::ucxx::sidl::rmi::NetworkException
      );



      /**
       * Create an new connection linked to an already existing object on a remote 
       * server.  The server and port number are in the url, the objectID is the unique ID
       * of the remote object in the remote instance registry. 
       * Return nil if protocol unknown or InstanceHandle.init() failed.
       */
      static ::ucxx::sidl::rmi::InstanceHandle
      connectInstance (
        /* in */const ::std::string& url
      )
      throw ( 
        ::ucxx::sidl::rmi::NetworkException
      );



      /**
       * <p>
       * Add one to the intrinsic reference count in the underlying object.
       * Object in <code>sidl</code> have an intrinsic reference count.
       * Objects continue to exist as long as the reference count is
       * positive. Clients should call this method whenever they
       * create another ongoing reference to an object or interface.
       * </p>
       * <p>
       * This does not have a return value because there is no language
       * independent type that can refer to an interface or a
       * class.
       * </p>
       */
      inline void
      addRef() throw () 
      {

        if ( !d_weak_reference ) {
          ior_t* loc_self = _get_ior();
          /*pack args to dispatch to ior*/
          (*(loc_self->d_epv->f_addRef))(loc_self );
          /*dispatch to ior*/
          /*unpack results and cleanup*/
        }
      }



      /**
       * Decrease by one the intrinsic reference count in the underlying
       * object, and delete the object if the reference is non-positive.
       * Objects in <code>sidl</code> have an intrinsic reference count.
       * Clients should call this method whenever they remove a
       * reference to an object or interface.
       */
      inline void
      deleteRef() throw () 
      {

        if ( !d_weak_reference ) {
          ior_t* loc_self = _get_ior();
          /*pack args to dispatch to ior*/
          (*(loc_self->d_epv->f_deleteRef))(loc_self );
          /*dispatch to ior*/
          /*unpack results and cleanup*/
          d_self = 0;
        }
      }



      /**
       * Return true if and only if <code>obj</code> refers to the same
       * object as this object.
       */
      bool
      isSame (
        /* in */::ucxx::sidl::BaseInterface iobj
      )
      throw () 
      ;



      /**
       * Check whether the object can support the specified interface or
       * class.  If the <code>sidl</code> type name in <code>name</code>
       * is supported, then a reference to that object is returned with the
       * reference count incremented.  The callee will be responsible for
       * calling <code>deleteRef</code> on the returned object.  If
       * the specified type is not supported, then a null reference is
       * returned.
       */
      ::ucxx::sidl::BaseInterface
      queryInt (
        /* in */const ::std::string& name
      )
      throw () 
      ;



      /**
       * Return whether this object is an instance of the specified type.
       * The string name must be the <code>sidl</code> type name.  This
       * routine will return <code>true</code> if and only if a cast to
       * the string type name would succeed.
       */
      bool
      isType (
        /* in */const ::std::string& name
      )
      throw () 
      ;



      /**
       * Return the meta-data about the class implementing this interface.
       */
      ::ucxx::sidl::ClassInfo
      getClassInfo() throw () 
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
      static ::ucxx::sidl::rmi::ProtocolFactory _create();

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

      ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

      const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); 
        }

      void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

      bool _is_nil() const { return (d_self==0); }

      bool _not_nil() const { return (d_self!=0); }

      bool operator !() const { return (d_self==0); }

      static inline const char * type_name() { return 
        "sidl.rmi.ProtocolFactory";}
      virtual void* _cast(const char* type) const;

    protected:
        // Pointer to external (DLL loadable) symbols (shared among instances)
        static const ext_t * s_ext;

      public:
        static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException 
          );

        static const sepv_t * _get_sepv() {
          return (*(_get_ext()->getStaticEPV))();
        }

      }; // end class ProtocolFactory
    } // end namespace rmi
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::sidl::rmi::ProtocolFactory > {
      typedef array< ::ucxx::sidl::rmi::ProtocolFactory > cxx_array_t;
      typedef ::ucxx::sidl::rmi::ProtocolFactory cxx_item_t;
      typedef struct sidl_rmi_ProtocolFactory__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct sidl_rmi_ProtocolFactory__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::sidl::rmi::ProtocolFactory > > 
        iterator;
      typedef const_array_iter< array_traits< 
        ::ucxx::sidl::rmi::ProtocolFactory > > const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::sidl::rmi::ProtocolFactory >: public interface_array< 
      array_traits< ::ucxx::sidl::rmi::ProtocolFactory > > {
    public:
      typedef interface_array< array_traits< ::ucxx::sidl::rmi::ProtocolFactory 
        > > Base;
      typedef array_traits< ::ucxx::sidl::rmi::ProtocolFactory >::cxx_array_t   
        cxx_array_t;
      typedef array_traits< ::ucxx::sidl::rmi::ProtocolFactory >::cxx_item_t    
        cxx_item_t;
      typedef array_traits< ::ucxx::sidl::rmi::ProtocolFactory >::ior_array_t   
        ior_array_t;
      typedef array_traits< ::ucxx::sidl::rmi::ProtocolFactory 
        >::ior_array_internal_t ior_array_internal_t;
      typedef array_traits< ::ucxx::sidl::rmi::ProtocolFactory >::ior_item_t    
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct sidl_rmi_ProtocolFactory__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::sidl::rmi::ProtocolFactory >&src) : Base(src) 
        {}

      /**
       * assignment
       */
      array< ::ucxx::sidl::rmi::ProtocolFactory >&
      operator =( const array< ::ucxx::sidl::rmi::ProtocolFactory >&rhs ) { 
        if (d_array != rhs._get_baseior()) {
          if (d_array) deleteRef();
          d_array = const_cast<sidl__array *>(rhs._get_baseior());
          if (d_array) addRef();
        }
        return *this;
      }

    };
  }

} //closes ucxx Namespace
#endif
