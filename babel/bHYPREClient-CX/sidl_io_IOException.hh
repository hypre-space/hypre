// 
// File:          sidl_io_IOException.hh
// Symbol:        sidl.io.IOException-v0.9.3
// Symbol Type:   class
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.io.IOException
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
// xml-url       = /home/painter/babel/share/babel-0.10.12/repository/sidl.io.IOException-v0.9.3.xml
// 

#ifndef included_sidl_io_IOException_hh
#define included_sidl_io_IOException_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace sidl { 
    namespace io { 

      class IOException;
    } // end namespace io
  } // end namespace sidl
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::sidl::io::IOException >;
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

#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
#ifndef included_sidl_io_IOException_IOR_h
#include "sidl_io_IOException_IOR.h"
#endif
#ifndef included_sidl_SIDLException_hh
#include "sidl_SIDLException.hh"
#endif

namespace ucxx { 
  namespace sidl { 
    namespace io { 

      /**
       * Symbol "sidl.io.IOException" (version 0.9.3)
       * 
       * generic exception for I/O issues 
       */
      class IOException: public virtual ::ucxx::sidl::SIDLException {

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
       * Return the message associated with the exception.
       */
      ::std::string
      getNote() throw () 
      ;


      /**
       * Set the message associated with the exception.
       */
      void
      setNote (
        /* in */const ::std::string& message
      )
      throw () 
      ;



      /**
       * Returns formatted string containing the concatenation of all 
       * tracelines.
       */
      ::std::string
      getTrace() throw () 
      ;


      /**
       * Adds a stringified entry/line to the stack trace.
       */
      void
      add (
        /* in */const ::std::string& traceline
      )
      throw () 
      ;



      /**
       * Formats and adds an entry to the stack trace based on the 
       * file name, line number, and method name.
       */
      void
      add (
        /* in */const ::std::string& filename,
        /* in */int32_t lineno,
        /* in */const ::std::string& methodname
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
      typedef struct sidl_io_IOException__object ior_t;
      typedef struct sidl_io_IOException__external ext_t;
      typedef struct sidl_io_IOException__sepv sepv_t;

      // default constructor
      IOException() { }

      // static constructor
      static ::ucxx::sidl::io::IOException _create();

      // default destructor
      virtual ~IOException () { }

      // copy constructor
      IOException ( const IOException& original );

      // assignment operator
      IOException& operator= ( const IOException& rhs );

      // conversion from ior to C++ class
      IOException ( IOException::ior_t* ior );

      // Alternate constructor: does not call addRef()
      // (sets d_weak_reference=isWeak)
      // For internal use by Impls (fixes bug#275)
      IOException ( IOException::ior_t* ior, bool isWeak );

      ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

      const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); 
        }

      void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

      bool _is_nil() const { return (d_self==0); }

      bool _not_nil() const { return (d_self!=0); }

      bool operator !() const { return (d_self==0); }

      static inline const char * type_name() { return "sidl.io.IOException";}
      virtual void* _cast(const char* type) const;

    protected:
        // Pointer to external (DLL loadable) symbols (shared among instances)
        static const ext_t * s_ext;

      public:
        static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException 
          );

      }; // end class IOException
    } // end namespace io
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::sidl::io::IOException > {
      typedef array< ::ucxx::sidl::io::IOException > cxx_array_t;
      typedef ::ucxx::sidl::io::IOException cxx_item_t;
      typedef struct sidl_io_IOException__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct sidl_io_IOException__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::sidl::io::IOException > > 
        iterator;
      typedef const_array_iter< array_traits< ::ucxx::sidl::io::IOException > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::sidl::io::IOException >: public interface_array< 
      array_traits< ::ucxx::sidl::io::IOException > > {
    public:
      typedef interface_array< array_traits< ::ucxx::sidl::io::IOException > > 
        Base;
      typedef array_traits< ::ucxx::sidl::io::IOException >::cxx_array_t        
        cxx_array_t;
      typedef array_traits< ::ucxx::sidl::io::IOException >::cxx_item_t         
        cxx_item_t;
      typedef array_traits< ::ucxx::sidl::io::IOException >::ior_array_t        
        ior_array_t;
      typedef array_traits< ::ucxx::sidl::io::IOException 
        >::ior_array_internal_t ior_array_internal_t;
      typedef array_traits< ::ucxx::sidl::io::IOException >::ior_item_t         
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct sidl_io_IOException__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::sidl::io::IOException >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::sidl::io::IOException >&
      operator =( const array< ::ucxx::sidl::io::IOException >&rhs ) { 
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
