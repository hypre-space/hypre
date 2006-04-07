// 
// File:          sidl_Resolve.hh
// Symbol:        sidl.Resolve-v0.9.3
// Symbol Type:   enumeration
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.Resolve
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
// xml-url       = /home/painter/babel-0.10.12/bin/.././share/repository/sidl.Resolve-v0.9.3.xml
// 

#ifndef included_sidl_Resolve_hh
#define included_sidl_Resolve_hh


#include "sidl_ucxx.hh"
#include "sidl_Resolve_IOR.h"

namespace ucxx { 
  namespace sidl { 

    enum Resolve {
      /**
       * Resolve symbols on an as needed basis. 
       */
      Resolve_LAZY = 0,

      /**
       * Resolve all symbols at load time. 
       */
      Resolve_NOW = 1,

      /**
       * Use the resolve setting from the SCL file. 
       */
      Resolve_SCLRESOLVE = 2

    };

  } // end namespace sidl
} // end namespace ucxx


struct sidl_Resolve__array;
namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::sidl::Resolve > {
      typedef array< ::ucxx::sidl::Resolve > cxx_array_t;
      typedef ::ucxx::sidl::Resolve cxx_item_t;
      typedef struct sidl_Resolve__array ior_array_t;
      typedef sidl_int__array ior_array_internal_t;
      typedef enum sidl_Resolve__enum ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type& reference;
      typedef value_type* pointer;
      typedef const value_type& const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::sidl::Resolve > > iterator;
      typedef const_array_iter< array_traits< ::ucxx::sidl::Resolve > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::sidl::Resolve >: public enum_array< array_traits< 
      ::ucxx::sidl::Resolve > > {
    public:
      typedef enum_array< array_traits< ::ucxx::sidl::Resolve > > Base;
      typedef array_traits< ::ucxx::sidl::Resolve >::cxx_array_t          
        cxx_array_t;
      typedef array_traits< ::ucxx::sidl::Resolve >::cxx_item_t           
        cxx_item_t;
      typedef array_traits< ::ucxx::sidl::Resolve >::ior_array_t          
        ior_array_t;
      typedef array_traits< ::ucxx::sidl::Resolve >::ior_array_internal_t 
        ior_array_internal_t;
      typedef array_traits< ::ucxx::sidl::Resolve >::ior_item_t           
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct sidl_Resolve__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::sidl::Resolve > &src) {
        d_array = src.d_array;
        if (d_array) addRef();
      }

      /**
       * assignment
       */
      array< ::ucxx::sidl::Resolve >&
      operator =( const array< ::ucxx::sidl::Resolve > &rhs) {
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
