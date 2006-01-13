// 
// File:          sidl_rmi_Response.hh
// Symbol:        sidl.rmi.Response-v0.9.3
// Symbol Type:   interface
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.rmi.Response
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
// xml-url       = /home/painter/babel/share/babel-0.10.12/repository/sidl.rmi.Response-v0.9.3.xml
// 

#ifndef included_sidl_rmi_Response_hh
#define included_sidl_rmi_Response_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace sidl { 
    namespace rmi { 

      class Response;
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
    class array< ::ucxx::sidl::rmi::Response >;
  }
} //closes ucxx Namespace
// 
// Forward declarations for method dependencies.
// 
namespace ucxx { 
  namespace sidl { 

    class BaseException;
  } // end namespace sidl
} // end namespace ucxx

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
    namespace io { 

      class IOException;
    } // end namespace io
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
#ifndef included_sidl_rmi_Response_IOR_h
#include "sidl_rmi_Response_IOR.h"
#endif
#ifndef included_sidl_io_Deserializer_hh
#include "sidl_io_Deserializer.hh"
#endif
#ifndef included_sidl_io_IOException_hh
#include "sidl_io_IOException.hh"
#endif
#ifndef included_sidl_rmi_NetworkException_hh
#include "sidl_rmi_NetworkException.hh"
#endif

namespace ucxx { 
  namespace sidl { 
    namespace rmi { 

      /**
       * Symbol "sidl.rmi.Response" (version 0.9.3)
       * 
       * This type is created when an InvocationHandle actually invokes its method.
       * It encapsulates all the results that users will want to pull out of a
       * remote method invocation.
       */
      class Response: public virtual ::ucxx::sidl::io::Deserializer {

      //////////////////////////////////////////////////
      // 
      // User Defined Methods
      // 

    public:

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
          ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
            sidl_BaseInterface__object * > 
            (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
            sidl_BaseInterface__object * > (d_self))->d_object,
            "sidl.rmi.Response");
          /*pack args to dispatch to ior*/
          (*(loc_self->d_epv->f_addRef))(loc_self->d_object );
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
          ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
            sidl_BaseInterface__object * > 
            (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
            sidl_BaseInterface__object * > (d_self))->d_object,
            "sidl.rmi.Response");
          /*pack args to dispatch to ior*/
          (*(loc_self->d_epv->f_deleteRef))(loc_self->d_object );
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

      /**
       * user defined non-static method.
       */
      void
      unpackBool (
        /* in */const ::std::string& key,
        /* out */bool& value
      )
      throw ( 
        ::ucxx::sidl::io::IOException
      );


      /**
       * user defined non-static method.
       */
      void
      unpackChar (
        /* in */const ::std::string& key,
        /* out */char& value
      )
      throw ( 
        ::ucxx::sidl::io::IOException
      );


      /**
       * user defined non-static method.
       */
      void
      unpackInt (
        /* in */const ::std::string& key,
        /* out */int32_t& value
      )
      throw ( 
        ::ucxx::sidl::io::IOException
      );


      /**
       * user defined non-static method.
       */
      void
      unpackLong (
        /* in */const ::std::string& key,
        /* out */int64_t& value
      )
      throw ( 
        ::ucxx::sidl::io::IOException
      );


      /**
       * user defined non-static method.
       */
      void
      unpackFloat (
        /* in */const ::std::string& key,
        /* out */float& value
      )
      throw ( 
        ::ucxx::sidl::io::IOException
      );


      /**
       * user defined non-static method.
       */
      void
      unpackDouble (
        /* in */const ::std::string& key,
        /* out */double& value
      )
      throw ( 
        ::ucxx::sidl::io::IOException
      );


      /**
       * user defined non-static method.
       */
      void
      unpackFcomplex (
        /* in */const ::std::string& key,
        /* out */::std::complex<float>& value
      )
      throw ( 
        ::ucxx::sidl::io::IOException
      );


      /**
       * user defined non-static method.
       */
      void
      unpackDcomplex (
        /* in */const ::std::string& key,
        /* out */::std::complex<double>& value
      )
      throw ( 
        ::ucxx::sidl::io::IOException
      );


      /**
       * user defined non-static method.
       */
      void
      unpackString (
        /* in */const ::std::string& key,
        /* out */::std::string& value
      )
      throw ( 
        ::ucxx::sidl::io::IOException
      );



      /**
       * if returns null, then safe to unpack arguments 
       */
      ::ucxx::sidl::BaseException
      getExceptionThrown() throw ( 
        ::ucxx::sidl::rmi::NetworkException
      );


      /**
       * signal that all is complete 
       */
      bool
      done() throw ( 
        ::ucxx::sidl::rmi::NetworkException
      );


      //////////////////////////////////////////////////
      // 
      // End User Defined Methods
      // (everything else in this file is specific to
      //  Babel's C++ bindings)
      // 

    public:
      typedef struct sidl_rmi_Response__object ior_t;
      typedef struct sidl_rmi_Response__external ext_t;
      typedef struct sidl_rmi_Response__sepv sepv_t;

      // default constructor
      Response() { }

      // default destructor
      virtual ~Response () { }

      // copy constructor
      Response ( const Response& original );

      // assignment operator
      Response& operator= ( const Response& rhs );

      // conversion from ior to C++ class
      Response ( Response::ior_t* ior );

      // Alternate constructor: does not call addRef()
      // (sets d_weak_reference=isWeak)
      // For internal use by Impls (fixes bug#275)
      Response ( Response::ior_t* ior, bool isWeak );

      ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

      const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); 
        }

      void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

      bool _is_nil() const { return (d_self==0); }

      bool _not_nil() const { return (d_self!=0); }

      bool operator !() const { return (d_self==0); }

      static inline const char * type_name() { return "sidl.rmi.Response";}
      virtual void* _cast(const char* type) const;

    protected:
        // Pointer to external (DLL loadable) symbols (shared among instances)
        static const ext_t * s_ext;

      public:
        static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException 
          );

      }; // end class Response
    } // end namespace rmi
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::sidl::rmi::Response > {
      typedef array< ::ucxx::sidl::rmi::Response > cxx_array_t;
      typedef ::ucxx::sidl::rmi::Response cxx_item_t;
      typedef struct sidl_rmi_Response__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct sidl_rmi_Response__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::sidl::rmi::Response > > 
        iterator;
      typedef const_array_iter< array_traits< ::ucxx::sidl::rmi::Response > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::sidl::rmi::Response >: public interface_array< 
      array_traits< ::ucxx::sidl::rmi::Response > > {
    public:
      typedef interface_array< array_traits< ::ucxx::sidl::rmi::Response > > 
        Base;
      typedef array_traits< ::ucxx::sidl::rmi::Response >::cxx_array_t          
        cxx_array_t;
      typedef array_traits< ::ucxx::sidl::rmi::Response >::cxx_item_t           
        cxx_item_t;
      typedef array_traits< ::ucxx::sidl::rmi::Response >::ior_array_t          
        ior_array_t;
      typedef array_traits< ::ucxx::sidl::rmi::Response >::ior_array_internal_t 
        ior_array_internal_t;
      typedef array_traits< ::ucxx::sidl::rmi::Response >::ior_item_t           
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct sidl_rmi_Response__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::sidl::rmi::Response >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::sidl::rmi::Response >&
      operator =( const array< ::ucxx::sidl::rmi::Response >&rhs ) { 
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
