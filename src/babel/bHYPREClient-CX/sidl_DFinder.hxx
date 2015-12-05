// 
// File:          sidl_DFinder.hxx
// Symbol:        sidl.DFinder-v0.9.15
// Symbol Type:   class
// Babel Version: 1.0.0
// Release:       $Name:  $
// Revision:      @(#) $Id: sidl_DFinder.hxx,v 1.2 2006/09/14 21:52:14 painter Exp $
// Description:   Client-side glue code for sidl.DFinder
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

#ifndef included_sidl_DFinder_hxx
#define included_sidl_DFinder_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace sidl { 

  class DFinder;
} // end namespace sidl

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::sidl::DFinder >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class DLL;
} // end namespace sidl

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_sidl_DFinder_IOR_h
#include "sidl_DFinder_IOR.h"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
#ifndef included_sidl_Finder_hxx
#include "sidl_Finder.hxx"
#endif
#ifndef included_sidl_Resolve_hxx
#include "sidl_Resolve.hxx"
#endif
#ifndef included_sidl_Scope_hxx
#include "sidl_Scope.hxx"
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
   * Symbol "sidl.DFinder" (version 0.9.15)
   * 
   * This class is the Default Finder.  If no Finder is set in class Loader,
   * this finder is used.  It uses SCL files from the filesystem to
   * resolve dynamic libraries.
   * 
   * The initial search path is taken from the SIDL_DLL_PATH
   * environment variable.
   */
  class DFinder: public virtual ::sidl::BaseClass,
    public virtual ::sidl::Finder {

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
     * Find a DLL containing the specified information for a sidl
     * class. This method searches through the files in set set path
     * looking for a shared library that contains the client-side or IOR
     * for a particular sidl class.
     * 
     * @param sidl_name  the fully qualified (long) name of the
     * class/interface to be found. Package names
     * are separated by period characters from each
     * other and the class/interface name.
     * @param target     to find a client-side binding, this is
     * normally the name of the language.
     * To find the implementation of a class
     * in order to make one, you should pass
     * the string "ior/impl" here.
     * @param lScope     this specifies whether the symbols should
     * be loaded into the global scope, a local
     * scope, or use the setting in the file.
     * @param lResolve   this specifies whether symbols should be
     * resolved as needed (LAZY), completely
     * resolved at load time (NOW), or use the
     * setting from the file.
     * @return a non-NULL object means the search was successful.
     * The DLL has already been added.
     */
    ::sidl::DLL
    findLibrary (
      /* in */const ::std::string& sidl_name,
      /* in */const ::std::string& target,
      /* in */::sidl::Scope lScope,
      /* in */::sidl::Resolve lResolve
    )
    ;



    /**
     * Set the search path, which is a semi-colon separated sequence of
     * URIs as described in class <code>DLL</code>.  This method will
     * invalidate any existing search path.
     */
    void
    setSearchPath (
      /* in */const ::std::string& path_name
    )
    ;



    /**
     * Return the current search path.  If the search path has not been
     * set, then the search path will be taken from environment variable
     * SIDL_DLL_PATH.
     */
    ::std::string
    getSearchPath() ;


    /**
     * Append the specified path fragment to the beginning of the
     * current search path.  If the search path has not yet been set
     * by a call to <code>setSearchPath</code>, then this fragment will
     * be appended to the path in environment variable SIDL_DLL_PATH.
     */
    void
    addSearchPath (
      /* in */const ::std::string& path_fragment
    )
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct sidl_DFinder__object ior_t;
    typedef struct sidl_DFinder__external ext_t;
    typedef struct sidl_DFinder__sepv sepv_t;

    // default constructor
    DFinder() { }

    // static constructor
    static ::sidl::DFinder _create();

    // RMI constructor
    static ::sidl::DFinder _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::sidl::DFinder _connect( /*in*/ const std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::sidl::DFinder _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~DFinder () { }

    // copy constructor
    DFinder ( const DFinder& original );

    // assignment operator
    DFinder& operator= ( const DFinder& rhs );

    // conversion from ior to C++ class
    DFinder ( DFinder::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    DFinder ( DFinder::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return "sidl.DFinder";}

    static struct sidl_DFinder__object* _cast(const void* src);

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

  }; // end class DFinder
} // end namespace sidl

extern "C" {


  #pragma weak sidl_DFinder__connectI

  #pragma weak sidl_DFinder__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct sidl_DFinder__object*
  sidl_DFinder__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct sidl_DFinder__object*
  sidl_DFinder__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::DFinder > {
    typedef array< ::sidl::DFinder > cxx_array_t;
    typedef ::sidl::DFinder cxx_item_t;
    typedef struct sidl_DFinder__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct sidl_DFinder__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::DFinder > > iterator;
    typedef const_array_iter< array_traits< ::sidl::DFinder > > const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::DFinder >: public interface_array< array_traits< 
    ::sidl::DFinder > > {
  public:
    typedef interface_array< array_traits< ::sidl::DFinder > > Base;
    typedef array_traits< ::sidl::DFinder >::cxx_array_t          cxx_array_t;
    typedef array_traits< ::sidl::DFinder >::cxx_item_t           cxx_item_t;
    typedef array_traits< ::sidl::DFinder >::ior_array_t          ior_array_t;
    typedef array_traits< ::sidl::DFinder >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::DFinder >::ior_item_t           ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_DFinder__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::DFinder >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::sidl::DFinder >&
    operator =( const array< ::sidl::DFinder >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#ifndef included_sidl_DLL_hxx
#include "sidl_DLL.hxx"
#endif
#endif
