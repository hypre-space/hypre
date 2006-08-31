// 
// File:          sidl_Resolve.hxx
// Symbol:        sidl.Resolve-v0.9.15
// Symbol Type:   enumeration
// Babel Version: 1.0.0
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

#ifndef included_sidl_Resolve_hxx
#define included_sidl_Resolve_hxx


#include "sidl_cxx.hxx"
#include "sidl_Resolve_IOR.h"

namespace sidl { 

  enum Resolve {
    Resolve_LAZY = 0,
    Resolve_NOW = 1,
    Resolve_SCLRESOLVE = 2
  };

} // end namespace sidl


struct sidl_Resolve__array;
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::Resolve > {
    typedef array< ::sidl::Resolve > cxx_array_t;
    typedef ::sidl::Resolve cxx_item_t;
    typedef struct sidl_Resolve__array ior_array_t;
    typedef sidl_int__array ior_array_internal_t;
    typedef enum sidl_Resolve__enum ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef const value_type& const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::Resolve > > iterator;
    typedef const_array_iter< array_traits< ::sidl::Resolve > > const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::Resolve >: public enum_array< array_traits< 
    ::sidl::Resolve > > {
  public:
    typedef enum_array< array_traits< ::sidl::Resolve > > Base;
    typedef array_traits< ::sidl::Resolve >::cxx_array_t          cxx_array_t;
    typedef array_traits< ::sidl::Resolve >::cxx_item_t           cxx_item_t;
    typedef array_traits< ::sidl::Resolve >::ior_array_t          ior_array_t;
    typedef array_traits< ::sidl::Resolve >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::Resolve >::ior_item_t           ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_Resolve__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::Resolve > &src) {
      d_array = src.d_array;
      if (d_array) addRef();
    }

    /**
     * assignment
     */
    array< ::sidl::Resolve >&
    operator =( const array< ::sidl::Resolve > &rhs) {
      if (d_array != rhs.d_array) {
        if (d_array) deleteRef();
        d_array = rhs.d_array;
        if (d_array) addRef();
      }
      return *this;
    }

  };
}


#endif
