// 
// File:          sidl_DLL.hxx
// Symbol:        sidl.DLL-v0.9.15
// Symbol Type:   class
// Babel Version: 1.0.4
// Release:       $Name: V2-4-0b $
// Revision:      @(#) $Id: sidl_DLL.hxx,v 1.4 2007/09/27 19:55:45 painter Exp $
// Description:   Client-side glue code for sidl.DLL
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

#ifndef included_sidl_DLL_hxx
#define included_sidl_DLL_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace sidl { 

  class DLL;
} // end namespace sidl

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::sidl::DLL >;
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
#ifndef included_sidl_DLL_IOR_h
#include "sidl_DLL_IOR.h"
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

  /**
   * Symbol "sidl.DLL" (version 0.9.15)
   * 
   * The <code>DLL</code> class encapsulates access to a single
   * dynamically linked library.  DLLs are loaded at run-time using
   * the <code>loadLibrary</code> method and later unloaded using
   * <code>unloadLibrary</code>.  Symbols in a loaded library are
   * resolved to an opaque pointer by method <code>lookupSymbol</code>.
   * Class instances are created by <code>createClass</code>.
   */
  class DLL: public virtual ::sidl::BaseClass {

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
     * Load a dynamic link library using the specified URI.  The
     * URI may be of the form "main:", "lib:", "file:", "ftp:", or
     * "http:".  A URI that starts with any other protocol string
     * is assumed to be a file name.  The "main:" URI creates a
     * library that allows access to global symbols in the running
     * program's main address space.  The "lib:X" URI converts the
     * library "X" into a platform-specific name (e.g., libX.so) and
     * loads that library.  The "file:" URI opens the DLL from the
     * specified file path.  The "ftp:" and "http:" URIs copy the
     * specified library from the remote site into a local temporary
     * file and open that file.  This method returns true if the
     * DLL was loaded successfully and false otherwise.  Note that
     * the "ftp:" and "http:" protocols are valid only if the W3C
     * WWW library is available.
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
     * now (at load time).
     */
    bool
    loadLibrary (
      /* in */const ::std::string& uri,
      /* in */bool loadGlobally,
      /* in */bool loadLazy
    )
    ;



    /**
     * Get the library name.  This is the name used to load the
     * library in <code>loadLibrary</code> except that all file names
     * contain the "file:" protocol.
     */
    ::std::string
    getName() ;


    /**
     * Return true if the library was loaded into the global namespace.
     */
    bool
    isGlobal() ;


    /**
     * Return true if the library was loaded using lazy symbol resolution.
     */
    bool
    isLazy() ;


    /**
     * Unload the dynamic link library.  The library may no longer
     * be used to access symbol names.  When the library is actually
     * unloaded from the memory image depends on details of the operating
     * system.
     */
    void
    unloadLibrary() ;


    /**
     * Lookup a symbol from the DLL and return the associated pointer.
     * A null value is returned if the name does not exist.
     */
    void*
    lookupSymbol (
      /* in */const ::std::string& linker_name
    )
    ;



    /**
     * Create an instance of the sidl class.  If the class constructor
     * is not defined in this DLL, then return null.
     */
    ::sidl::BaseClass
    createClass (
      /* in */const ::std::string& sidl_name
    )
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct sidl_DLL__object ior_t;
    typedef struct sidl_DLL__external ext_t;
    typedef struct sidl_DLL__sepv sepv_t;

    // default constructor
    DLL() { 
    }

    // static constructor
    static ::sidl::DLL _create();

    // RMI constructor
    static ::sidl::DLL _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::sidl::DLL _connect( /*in*/ const std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::sidl::DLL _connect( /*in*/ const std::string& url, /*in*/ const 
      bool ar  );

    // default destructor
    virtual ~DLL () { }

    // copy constructor
    DLL ( const DLL& original );

    // assignment operator
    DLL& operator= ( const DLL& rhs );

    // conversion from ior to C++ class
    DLL ( DLL::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    DLL ( DLL::ior_t* ior, bool isWeak );

    inline ior_t* _get_ior() const throw() {
      return reinterpret_cast< ior_t*>(d_self);
    }

    void _set_ior( ior_t* ptr ) throw () { 
      d_self = reinterpret_cast< void*>(ptr);
    }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return "sidl.DLL";}

    static struct sidl_DLL__object* _cast(const void* src);

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

  }; // end class DLL
} // end namespace sidl

extern "C" {


#pragma weak sidl_DLL__connectI

#pragma weak sidl_DLL__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct sidl_DLL__object*
  sidl_DLL__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct sidl_DLL__object*
  sidl_DLL__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::DLL > {
    typedef array< ::sidl::DLL > cxx_array_t;
    typedef ::sidl::DLL cxx_item_t;
    typedef struct sidl_DLL__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct sidl_DLL__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::DLL > > iterator;
    typedef const_array_iter< array_traits< ::sidl::DLL > > const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::DLL >: public interface_array< array_traits< ::sidl::DLL 
    > > {
  public:
    typedef interface_array< array_traits< ::sidl::DLL > > Base;
    typedef array_traits< ::sidl::DLL >::cxx_array_t          cxx_array_t;
    typedef array_traits< ::sidl::DLL >::cxx_item_t           cxx_item_t;
    typedef array_traits< ::sidl::DLL >::ior_array_t          ior_array_t;
    typedef array_traits< ::sidl::DLL >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::DLL >::ior_item_t           ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_DLL__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::DLL >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::sidl::DLL >&
    operator =( const array< ::sidl::DLL >&rhs ) { 
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
