// 
// File:          sidl_Loader.hxx
// Symbol:        sidl.Loader-v0.9.15
// Symbol Type:   class
// Babel Version: 1.0.0
// Release:       $Name:  $
// Revision:      @(#) $Id: sidl_Loader.hxx,v 1.2 2006/09/14 21:52:15 painter Exp $
// Description:   Client-side glue code for sidl.Loader
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

#ifndef included_sidl_Loader_hxx
#define included_sidl_Loader_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace sidl { 

  class Loader;
} // end namespace sidl

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::sidl::Loader >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class DLL;
} // end namespace sidl

namespace sidl { 

  class Finder;
} // end namespace sidl

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_sidl_Loader_IOR_h
#include "sidl_Loader_IOR.h"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
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
   * Symbol "sidl.Loader" (version 0.9.15)
   * 
   * Class <code>Loader</code> manages dyanamic loading and symbol name
   * resolution for the sidl runtime system.  The <code>Loader</code> class
   * manages a library search path and keeps a record of all libraries
   * loaded through this interface, including the initial "global" symbols
   * in the main program.
   * 
   * Unless explicitly set, the <code>Loader</code> uses the default
   * <code>sidl.Finder</code> implemented in <code>sidl.DFinder</code>.
   * This class searches the filesystem for <code>.scl</code> files when
   * trying to find a class. The initial path is taken from the
   * environment variable SIDL_DLL_PATH, which is a semi-colon
   * separated sequence of URIs as described in class <code>DLL</code>.
   */
  class Loader: public virtual ::sidl::BaseClass {

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
     * Load the specified library if it has not already been loaded.
     * The URI format is defined in class <code>DLL</code>.  The search
     * path is not searched to resolve the library name.
     * 
     * @param uri          the URI to load. This can be a .la file
     * (a metadata file produced by libtool) or
     * a shared library binary (i.e., .so,
     * .dll or whatever is appropriate for your
     * OS)
     * @param loadGlobally <code>true</code> means that the shared
     * library symbols will be loaded into the
     * global namespace; <code>false</code> 
     * means they will be loaded into a 
     * private namespace. Some operating systems
     * may not be able to honor the value presented
     * here.
     * @param loadLazy     <code>true</code> instructs the loader to
     * that symbols can be resolved as needed (lazy)
     * instead of requiring everything to be resolved
     * now.
     * @return if the load was successful, a non-NULL DLL object is returned.
     */
    static ::sidl::DLL
    loadLibrary (
      /* in */const ::std::string& uri,
      /* in */bool loadGlobally,
      /* in */bool loadLazy
    )
    ;



    /**
     * Append the specified DLL to the beginning of the list of already
     * loaded DLLs.
     */
    static void
    addDLL (
      /* in */::sidl::DLL dll
    )
    ;



    /**
     * Unload all dynamic link libraries.  The library may no longer
     * be used to access symbol names.  When the library is actually
     * unloaded from the memory image depends on details of the operating
     * system.
     */
    static void
    unloadLibraries() ;


    /**
     * Find a DLL containing the specified information for a sidl
     * class. This method searches SCL files in the search path looking
     * for a shared library that contains the client-side or IOR
     * for a particular sidl class.
     * 
     * This call is implemented by calling the current
     * <code>Finder</code>. The default finder searches the local
     * file system for <code>.scl</code> files to locate the
     * target class/interface.
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
     * scope, or use the setting in the SCL file.
     * @param lResolve   this specifies whether symbols should be
     * resolved as needed (LAZY), completely
     * resolved at load time (NOW), or use the
     * setting from the SCL file.
     * @return a non-NULL object means the search was successful.
     * The DLL has already been added.
     */
    static ::sidl::DLL
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
     * 
     * This updates the search path in the current <code>Finder</code>.
     */
    static void
    setSearchPath (
      /* in */const ::std::string& path_name
    )
    ;



    /**
     * Return the current search path.  The default
     * <code>Finder</code> initializes the search path
     * from environment variable SIDL_DLL_PATH.
     */
    static ::std::string
    getSearchPath() ;


    /**
     * Append the specified path fragment to the beginning of the
     * current search path.  This method operates on the Loader's
     * current <code>Finder</code>. This will add a path to the
     * current search path. Normally, the search path is initialized
     * from the SIDL_DLL_PATH environment variable.
     */
    static void
    addSearchPath (
      /* in */const ::std::string& path_fragment
    )
    ;



    /**
     * This method sets the <code>Finder</code> that
     * <code>Loader</code> will use to find DLLs.  If no
     * <code>Finder</code> is set or if NULL is passed in, the Default
     * Finder <code>DFinder</code> will be used.
     * 
     * Future calls to <code>findLibrary</code>,
     * <code>addSearchPath</code>, <code>getSearchPath</code>, and
     * <code>setSearchPath</code> are deligated to the
     * <code>Finder</code> set here.
     */
    static void
    setFinder (
      /* in */::sidl::Finder f
    )
    ;



    /**
     * This method gets the <code>Finder</code> that <code>Loader</code>
     * uses to find DLLs.  
     */
    static ::sidl::Finder
    getFinder() ;


    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct sidl_Loader__object ior_t;
    typedef struct sidl_Loader__external ext_t;
    typedef struct sidl_Loader__sepv sepv_t;

    // default constructor
    Loader() { }

    // static constructor
    static ::sidl::Loader _create();

    // RMI constructor
    static ::sidl::Loader _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::sidl::Loader _connect( /*in*/ const std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::sidl::Loader _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~Loader () { }

    // copy constructor
    Loader ( const Loader& original );

    // assignment operator
    Loader& operator= ( const Loader& rhs );

    // conversion from ior to C++ class
    Loader ( Loader::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    Loader ( Loader::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return "sidl.Loader";}

    static struct sidl_Loader__object* _cast(const void* src);

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

  }; // end class Loader
} // end namespace sidl

extern "C" {


  #pragma weak sidl_Loader__connectI

  #pragma weak sidl_Loader__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct sidl_Loader__object*
  sidl_Loader__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct sidl_Loader__object*
  sidl_Loader__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::Loader > {
    typedef array< ::sidl::Loader > cxx_array_t;
    typedef ::sidl::Loader cxx_item_t;
    typedef struct sidl_Loader__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct sidl_Loader__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::Loader > > iterator;
    typedef const_array_iter< array_traits< ::sidl::Loader > > const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::Loader >: public interface_array< array_traits< 
    ::sidl::Loader > > {
  public:
    typedef interface_array< array_traits< ::sidl::Loader > > Base;
    typedef array_traits< ::sidl::Loader >::cxx_array_t          cxx_array_t;
    typedef array_traits< ::sidl::Loader >::cxx_item_t           cxx_item_t;
    typedef array_traits< ::sidl::Loader >::ior_array_t          ior_array_t;
    typedef array_traits< ::sidl::Loader >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::Loader >::ior_item_t           ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_Loader__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::Loader >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::sidl::Loader >&
    operator =( const array< ::sidl::Loader >&rhs ) { 
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
#ifndef included_sidl_Finder_hxx
#include "sidl_Finder.hxx"
#endif
#endif
