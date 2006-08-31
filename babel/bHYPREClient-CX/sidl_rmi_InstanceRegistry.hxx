// 
// File:          sidl_rmi_InstanceRegistry.hxx
// Symbol:        sidl.rmi.InstanceRegistry-v0.9.15
// Symbol Type:   class
// Babel Version: 1.0.0
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.rmi.InstanceRegistry
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

#ifndef included_sidl_rmi_InstanceRegistry_hxx
#define included_sidl_rmi_InstanceRegistry_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace sidl { 
  namespace rmi { 

    class InstanceRegistry;
  } // end namespace rmi
} // end namespace sidl

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::sidl::rmi::InstanceRegistry >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class BaseClass;
} // end namespace sidl

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_IOR_h
#include "sidl_rmi_InstanceRegistry_IOR.h"
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
     * Symbol "sidl.rmi.InstanceRegistry" (version 0.9.15)
     * 
     *  
     * This singleton class is implemented by Babel's runtime for RMI
     * libraries to invoke methods on server objects.  It maps
     * objectID strings to sidl_BaseClass objects and vice-versa.
     * 
     * The InstanceRegistry creates and returns a unique string when a
     * new object is added to the registry.  When an object's refcount
     * reaches 0 and it is collected, it is removed from the Instance
     * Registry.
     * 
     * Objects are added to the registry in 3 ways:
     * 1) Added to the server's registry when an object is
     * create[Remote]'d.
     * 2) Implicity added to the local registry when an object is
     * passed as an argument in a remote call.
     * 3) A user may manually add a reference to the local registry
     * for publishing purposes.  The user hsould keep a reference
     * to the object.  Currently, the user cannot provide their own
     * objectID, this capability should probably be added.
     */
    class InstanceRegistry: public virtual ::sidl::BaseClass {

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
       * Register an instance of a class.
       * 
       * the registry will return an objectID string guaranteed to be
       * unique for the lifetime of the process
       */
      static ::std::string
      registerInstance (
        /* in */::sidl::BaseClass instance
      )
      ;



      /**
       *  
       * Register an instance of a class with the given instanceID
       * 
       * If a different object already exists in registry under
       * the supplied name, a false is returned, if the object was 
       * successfully registered, true is returned.
       */
      static ::std::string
      registerInstance (
        /* in */::sidl::BaseClass instance,
        /* in */const ::std::string& instanceID
      )
      ;



      /**
       *  
       * returns a handle to the class based on the unique objectID
       * string, (null if the handle isn't in the table)
       */
      static ::sidl::BaseClass
      getInstance (
        /* in */const ::std::string& instanceID
      )
      ;



      /**
       *  
       * takes a class and returns the objectID string associated
       * with it.  (null if the handle isn't in the table)
       */
      static ::std::string
      getInstance (
        /* in */::sidl::BaseClass instance
      )
      ;



      /**
       *  
       * removes an instance from the table based on its objectID
       * string..  returns a pointer to the object, which must be
       * destroyed.
       */
      static ::sidl::BaseClass
      removeInstance (
        /* in */const ::std::string& instanceID
      )
      ;



      /**
       *  
       * removes an instance from the table based on its BaseClass
       * pointer.  returns the objectID string, which much be freed.
       */
      static ::std::string
      removeInstance (
        /* in */::sidl::BaseClass instance
      )
      ;



      //////////////////////////////////////////////////
      // 
      // End User Defined Methods
      // (everything else in this file is specific to
      //  Babel's C++ bindings)
      // 

    public:
      typedef struct sidl_rmi_InstanceRegistry__object ior_t;
      typedef struct sidl_rmi_InstanceRegistry__external ext_t;
      typedef struct sidl_rmi_InstanceRegistry__sepv sepv_t;

      // default constructor
      InstanceRegistry() { }

      // static constructor
      static ::sidl::rmi::InstanceRegistry _create();

      // RMI constructor
      static ::sidl::rmi::InstanceRegistry _create( /*in*/ const std::string& 
        url );

      // RMI connect
      static inline ::sidl::rmi::InstanceRegistry _connect( /*in*/ const 
        std::string& url ) { 
        return _connect(url, true);
      }

      // RMI connect 2
      static ::sidl::rmi::InstanceRegistry _connect( /*in*/ const std::string& 
        url, /*in*/ const bool ar  );

      // default destructor
      virtual ~InstanceRegistry () { }

      // copy constructor
      InstanceRegistry ( const InstanceRegistry& original );

      // assignment operator
      InstanceRegistry& operator= ( const InstanceRegistry& rhs );

      // conversion from ior to C++ class
      InstanceRegistry ( InstanceRegistry::ior_t* ior );

      // Alternate constructor: does not call addRef()
      // (sets d_weak_reference=isWeak)
      // For internal use by Impls (fixes bug#275)
      InstanceRegistry ( InstanceRegistry::ior_t* ior, bool isWeak );

      ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

      const ior_t* _get_ior() const throw () { return reinterpret_cast< 
        ior_t*>(d_self); }

      void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
        void*>(ptr); }

      bool _is_nil() const throw () { return (d_self==0); }

      bool _not_nil() const throw () { return (d_self!=0); }

      bool operator !() const throw () { return (d_self==0); }

      static inline const char * type_name() throw () { return 
        "sidl.rmi.InstanceRegistry";}

      static struct sidl_rmi_InstanceRegistry__object* _cast(const void* src);

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

    }; // end class InstanceRegistry
  } // end namespace rmi
} // end namespace sidl

extern "C" {


  #pragma weak sidl_rmi_InstanceRegistry__connectI

  #pragma weak sidl_rmi_InstanceRegistry__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct sidl_rmi_InstanceRegistry__object*
  sidl_rmi_InstanceRegistry__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct sidl_rmi_InstanceRegistry__object*
  sidl_rmi_InstanceRegistry__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::rmi::InstanceRegistry > {
    typedef array< ::sidl::rmi::InstanceRegistry > cxx_array_t;
    typedef ::sidl::rmi::InstanceRegistry cxx_item_t;
    typedef struct sidl_rmi_InstanceRegistry__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct sidl_rmi_InstanceRegistry__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::rmi::InstanceRegistry > > 
      iterator;
    typedef const_array_iter< array_traits< ::sidl::rmi::InstanceRegistry > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::rmi::InstanceRegistry >: public interface_array< 
    array_traits< ::sidl::rmi::InstanceRegistry > > {
  public:
    typedef interface_array< array_traits< ::sidl::rmi::InstanceRegistry > > 
      Base;
    typedef array_traits< ::sidl::rmi::InstanceRegistry >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::sidl::rmi::InstanceRegistry >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::sidl::rmi::InstanceRegistry >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::sidl::rmi::InstanceRegistry >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::rmi::InstanceRegistry >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_rmi_InstanceRegistry__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::rmi::InstanceRegistry >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::sidl::rmi::InstanceRegistry >&
    operator =( const array< ::sidl::rmi::InstanceRegistry >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#endif
