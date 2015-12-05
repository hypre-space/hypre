// 
// File:          sidl_NotImplementedException.hxx
// Symbol:        sidl.NotImplementedException-v0.9.15
// Symbol Type:   class
// Babel Version: 1.0.4
// Release:       $Name: V2-4-0b $
// Revision:      @(#) $Id: sidl_NotImplementedException.hxx,v 1.4 2007/09/27 19:55:46 painter Exp $
// Description:   Client-side glue code for sidl.NotImplementedException
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

#ifndef included_sidl_NotImplementedException_hxx
#define included_sidl_NotImplementedException_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace sidl { 

  class NotImplementedException;
} // end namespace sidl

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::sidl::NotImplementedException >;
}
#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_sidl_NotImplementedException_IOR_h
#include "sidl_NotImplementedException_IOR.h"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif
#ifndef included_sidl_SIDLException_hxx
#include "sidl_SIDLException.hxx"
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
   * Symbol "sidl.NotImplementedException" (version 0.9.15)
   * 
   *  
   * This Exception is thrown when a method is called that an 
   * implmentation has not been written for yet.  The throw code is
   * placed into the _Impl files automatically when they are generated.
   */
  class NotImplementedException: public virtual ::sidl::RuntimeException, 
    public virtual ::sidl::SIDLException {

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

    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct sidl_NotImplementedException__object ior_t;
    typedef struct sidl_NotImplementedException__external ext_t;
    typedef struct sidl_NotImplementedException__sepv sepv_t;

    // default constructor
    NotImplementedException() { 
    }

    // static constructor
    static ::sidl::NotImplementedException _create();

    // RMI constructor
    static ::sidl::NotImplementedException _create( /*in*/ const std::string& 
      url );

    // RMI connect
    static inline ::sidl::NotImplementedException _connect( /*in*/ const 
      std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::sidl::NotImplementedException _connect( /*in*/ const std::string& 
      url, /*in*/ const bool ar  );

    // default destructor
    virtual ~NotImplementedException () { }

    // copy constructor
    NotImplementedException ( const NotImplementedException& original );

    // assignment operator
    NotImplementedException& operator= ( const NotImplementedException& rhs );

    // conversion from ior to C++ class
    NotImplementedException ( NotImplementedException::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    NotImplementedException ( NotImplementedException::ior_t* ior, bool isWeak 
      );

    inline ior_t* _get_ior() const throw() {
      return reinterpret_cast< ior_t*>(d_self);
    }

    void _set_ior( ior_t* ptr ) throw () { 
      d_self = reinterpret_cast< void*>(ptr);
    }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "sidl.NotImplementedException";}

    static struct sidl_NotImplementedException__object* _cast(const void* src);

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

  }; // end class NotImplementedException
} // end namespace sidl

extern "C" {


#pragma weak sidl_NotImplementedException__connectI

#pragma weak sidl_NotImplementedException__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct sidl_NotImplementedException__object*
  sidl_NotImplementedException__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct sidl_NotImplementedException__object*
  sidl_NotImplementedException__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::NotImplementedException > {
    typedef array< ::sidl::NotImplementedException > cxx_array_t;
    typedef ::sidl::NotImplementedException cxx_item_t;
    typedef struct sidl_NotImplementedException__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct sidl_NotImplementedException__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::NotImplementedException > > 
      iterator;
    typedef const_array_iter< array_traits< ::sidl::NotImplementedException > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::NotImplementedException >: public interface_array< 
    array_traits< ::sidl::NotImplementedException > > {
  public:
    typedef interface_array< array_traits< ::sidl::NotImplementedException > > 
      Base;
    typedef array_traits< ::sidl::NotImplementedException >::cxx_array_t        
      cxx_array_t;
    typedef array_traits< ::sidl::NotImplementedException >::cxx_item_t         
      cxx_item_t;
    typedef array_traits< ::sidl::NotImplementedException >::ior_array_t        
      ior_array_t;
    typedef array_traits< ::sidl::NotImplementedException 
      >::ior_array_internal_t ior_array_internal_t;
    typedef array_traits< ::sidl::NotImplementedException >::ior_item_t         
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_NotImplementedException__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::NotImplementedException >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::sidl::NotImplementedException >&
    operator =( const array< ::sidl::NotImplementedException >&rhs ) { 
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
