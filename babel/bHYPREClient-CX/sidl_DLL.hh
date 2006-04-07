// 
// File:          sidl_DLL.hh
// Symbol:        sidl.DLL-v0.9.3
// Symbol Type:   class
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
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
// babel-version = 0.10.12
// xml-url       = /home/painter/babel-0.10.12/bin/.././share/repository/sidl.DLL-v0.9.3.xml
// 

#ifndef included_sidl_DLL_hh
#define included_sidl_DLL_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace sidl { 

    class DLL;
  } // end namespace sidl
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::sidl::DLL >;
  }
} //closes ucxx Namespace
// 
// Forward declarations for method dependencies.
// 
namespace ucxx { 
  namespace sidl { 

    class BaseClass;
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

#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
#ifndef included_sidl_DLL_IOR_h
#include "sidl_DLL_IOR.h"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif

namespace ucxx { 
  namespace sidl { 

    /**
     * Symbol "sidl.DLL" (version 0.9.3)
     * 
     * The <code>DLL</code> class encapsulates access to a single
     * dynamically linked library.  DLLs are loaded at run-time using
     * the <code>loadLibrary</code> method and later unloaded using
     * <code>unloadLibrary</code>.  Symbols in a loaded library are
     * resolved to an opaque pointer by method <code>lookupSymbol</code>.
     * Class instances are created by <code>createClass</code>.
     */
    class DLL: public virtual ::ucxx::sidl::BaseClass {

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
     *                     (a metadata file produced by libtool) or
     *                     a shared library binary (i.e., .so,
     *                     .dll or whatever is appropriate for your
     *                     OS)
     * @param loadGlobally <code>true</code> means that the shared
     *                     library symbols will be loaded into the
     *                     global namespace; <code>false</code> 
     *                     means they will be loaded into a 
     *                     private namespace. Some operating systems
     *                     may not be able to honor the value presented
     *                     here.
     * @param loadLazy     <code>true</code> instructs the loader to
     *                     that symbols can be resolved as needed (lazy)
     *                     instead of requiring everything to be resolved
     *                     now (at load time).
     */
    bool
    loadLibrary (
      /* in */const ::std::string& uri,
      /* in */bool loadGlobally,
      /* in */bool loadLazy
    )
    throw () 
    ;



    /**
     * Get the library name.  This is the name used to load the
     * library in <code>loadLibrary</code> except that all file names
     * contain the "file:" protocol.
     */
    ::std::string
    getName() throw () 
    ;


    /**
     * Unload the dynamic link library.  The library may no longer
     * be used to access symbol names.  When the library is actually
     * unloaded from the memory image depends on details of the operating
     * system.
     */
    inline void
    unloadLibrary() throw () 
    {

      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      (*(loc_self->d_epv->f_unloadLibrary))(loc_self );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
    }



    /**
     * Lookup a symbol from the DLL and return the associated pointer.
     * A null value is returned if the name does not exist.
     */
    void*
    lookupSymbol (
      /* in */const ::std::string& linker_name
    )
    throw () 
    ;



    /**
     * Create an instance of the sidl class.  If the class constructor
     * is not defined in this DLL, then return null.
     */
    ::ucxx::sidl::BaseClass
    createClass (
      /* in */const ::std::string& sidl_name
    )
    throw () 
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
    DLL() { }

    // static constructor
    static ::ucxx::sidl::DLL _create();

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

    ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

    static inline const char * type_name() { return "sidl.DLL";}
    virtual void* _cast(const char* type) const;

  protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException );

    }; // end class DLL
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::sidl::DLL > {
      typedef array< ::ucxx::sidl::DLL > cxx_array_t;
      typedef ::ucxx::sidl::DLL cxx_item_t;
      typedef struct sidl_DLL__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct sidl_DLL__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::sidl::DLL > > iterator;
      typedef const_array_iter< array_traits< ::ucxx::sidl::DLL > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::sidl::DLL >: public interface_array< array_traits< 
      ::ucxx::sidl::DLL > > {
    public:
      typedef interface_array< array_traits< ::ucxx::sidl::DLL > > Base;
      typedef array_traits< ::ucxx::sidl::DLL >::cxx_array_t          
        cxx_array_t;
      typedef array_traits< ::ucxx::sidl::DLL >::cxx_item_t           
        cxx_item_t;
      typedef array_traits< ::ucxx::sidl::DLL >::ior_array_t          
        ior_array_t;
      typedef array_traits< ::ucxx::sidl::DLL >::ior_array_internal_t 
        ior_array_internal_t;
      typedef array_traits< ::ucxx::sidl::DLL >::ior_item_t           
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct sidl_DLL__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::sidl::DLL >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::sidl::DLL >&
      operator =( const array< ::ucxx::sidl::DLL >&rhs ) { 
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
