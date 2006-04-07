// 
// File:          sidl_Scope.hh
// Symbol:        sidl.Scope-v0.9.3
// Symbol Type:   enumeration
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.Scope
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
// xml-url       = /home/painter/babel-0.10.12/bin/.././share/repository/sidl.Scope-v0.9.3.xml
// 

#ifndef included_sidl_Scope_hh
#define included_sidl_Scope_hh


#include "sidl_ucxx.hh"
#include "sidl_Scope_IOR.h"

namespace ucxx { 
  namespace sidl { 

    enum Scope {
      /**
       * Attempt to load the symbols into a local namespace. 
       */
      Scope_LOCAL = 0,

      /**
       * Attempt to load the symbols into the global namespace. 
       */
      Scope_GLOBAL = 1,

      /**
       * Use the scope setting from the SCL file. 
       */
      Scope_SCLSCOPE = 2

    };

  } // end namespace sidl
} // end namespace ucxx


struct sidl_Scope__array;
namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::sidl::Scope > {
      typedef array< ::ucxx::sidl::Scope > cxx_array_t;
      typedef ::ucxx::sidl::Scope cxx_item_t;
      typedef struct sidl_Scope__array ior_array_t;
      typedef sidl_int__array ior_array_internal_t;
      typedef enum sidl_Scope__enum ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type& reference;
      typedef value_type* pointer;
      typedef const value_type& const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::sidl::Scope > > iterator;
      typedef const_array_iter< array_traits< ::ucxx::sidl::Scope > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::sidl::Scope >: public enum_array< array_traits< 
      ::ucxx::sidl::Scope > > {
    public:
      typedef enum_array< array_traits< ::ucxx::sidl::Scope > > Base;
      typedef array_traits< ::ucxx::sidl::Scope >::cxx_array_t          
        cxx_array_t;
      typedef array_traits< ::ucxx::sidl::Scope >::cxx_item_t           
        cxx_item_t;
      typedef array_traits< ::ucxx::sidl::Scope >::ior_array_t          
        ior_array_t;
      typedef array_traits< ::ucxx::sidl::Scope >::ior_array_internal_t 
        ior_array_internal_t;
      typedef array_traits< ::ucxx::sidl::Scope >::ior_item_t           
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct sidl_Scope__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::sidl::Scope > &src) {
        d_array = src.d_array;
        if (d_array) addRef();
      }

      /**
       * assignment
       */
      array< ::ucxx::sidl::Scope >&
      operator =( const array< ::ucxx::sidl::Scope > &rhs) {
        if (d_array != rhs.d_array) {
          if (d_array) deleteRef();
          d_array = rhs.d_array;
          if (d_array) addRef();
        }
        return *this;
      }

    };
  }

  }


  #endif
