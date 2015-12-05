// 
// File:          sidl_BaseClass.hxx
// Symbol:        sidl.BaseClass-v0.9.15
// Symbol Type:   class
// Babel Version: 1.0.0
// Release:       $Name: V1-14-0b $
// Revision:      @(#) $Id: sidl_BaseClass.hxx,v 1.2 2006/09/14 21:52:14 painter Exp $
// Description:   Client-side glue code for sidl.BaseClass
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

#ifndef included_sidl_BaseClass_hxx
#define included_sidl_BaseClass_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace sidl { 

  class BaseClass;
} // end namespace sidl

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::sidl::BaseClass >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class BaseInterface;
} // end namespace sidl

namespace sidl { 

  class ClassInfo;
} // end namespace sidl

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
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

  /**
   * Symbol "sidl.BaseClass" (version 0.9.15)
   * 
   * Every class implicitly inherits from <code>BaseClass</code>.  This
   * class implements the methods in <code>BaseInterface</code>.
   */
  class BaseClass: public virtual ::sidl::BaseInterface {

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
    void
    addRef() ;


    /**
     * Decrease by one the intrinsic reference count in the underlying
     * object, and delete the object if the reference is non-positive.
     * Objects in <code>sidl</code> have an intrinsic reference count.
     * Clients should call this method whenever they remove a
     * reference to an object or interface.
     */
    void
    deleteRef() ;


    /**
     * Return true if and only if <code>obj</code> refers to the same
     * object as this object.
     */
    bool
    isSame (
      /* in */::sidl::BaseInterface iobj
    )
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
    ;



    /**
     * Return the meta-data about the class implementing this interface.
     */
    ::sidl::ClassInfo
    getClassInfo() ;


    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct sidl_BaseClass__object ior_t;
    typedef struct sidl_BaseClass__external ext_t;
    typedef struct sidl_BaseClass__sepv sepv_t;

    // default constructor
    BaseClass() { }

    // static constructor
    static ::sidl::BaseClass _create();

    // RMI constructor
    static ::sidl::BaseClass _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::sidl::BaseClass _connect( /*in*/ const std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::sidl::BaseClass _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~BaseClass () { }

    // copy constructor
    BaseClass ( const BaseClass& original );

    // assignment operator
    BaseClass& operator= ( const BaseClass& rhs );

    // conversion from ior to C++ class
    BaseClass ( BaseClass::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    BaseClass ( BaseClass::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return "sidl.BaseClass";}

    static struct sidl_BaseClass__object* _cast(const void* src);

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

  }; // end class BaseClass
} // end namespace sidl

extern "C" {


  #pragma weak sidl_BaseClass__connectI

  #pragma weak sidl_BaseClass__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct sidl_BaseClass__object*
  sidl_BaseClass__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct sidl_BaseClass__object*
  sidl_BaseClass__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::BaseClass > {
    typedef array< ::sidl::BaseClass > cxx_array_t;
    typedef ::sidl::BaseClass cxx_item_t;
    typedef struct sidl_BaseClass__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct sidl_BaseClass__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::BaseClass > > iterator;
    typedef const_array_iter< array_traits< ::sidl::BaseClass > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::BaseClass >: public interface_array< array_traits< 
    ::sidl::BaseClass > > {
  public:
    typedef interface_array< array_traits< ::sidl::BaseClass > > Base;
    typedef array_traits< ::sidl::BaseClass >::cxx_array_t          cxx_array_t;
    typedef array_traits< ::sidl::BaseClass >::cxx_item_t           cxx_item_t;
    typedef array_traits< ::sidl::BaseClass >::ior_array_t          ior_array_t;
    typedef array_traits< ::sidl::BaseClass >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::BaseClass >::ior_item_t           ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_BaseClass__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::BaseClass >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::sidl::BaseClass >&
    operator =( const array< ::sidl::BaseClass >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#endif
