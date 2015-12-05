// 
// File:          sidl_Scope.hxx
// Symbol:        sidl.Scope-v0.9.15
// Symbol Type:   enumeration
// Babel Version: 1.0.0
// Release:       $Name:  $
// Revision:      @(#) $Id: sidl_Scope.hxx,v 1.2 2006/09/14 21:52:15 painter Exp $
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

#ifndef included_sidl_Scope_hxx
#define included_sidl_Scope_hxx


#include "sidl_cxx.hxx"
#include "sidl_Scope_IOR.h"

namespace sidl { 

  enum Scope {
    Scope_LOCAL = 0,
    Scope_GLOBAL = 1,
    Scope_SCLSCOPE = 2
  };

} // end namespace sidl


struct sidl_Scope__array;
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::sidl::Scope > {
    typedef array< ::sidl::Scope > cxx_array_t;
    typedef ::sidl::Scope cxx_item_t;
    typedef struct sidl_Scope__array ior_array_t;
    typedef sidl_int__array ior_array_internal_t;
    typedef enum sidl_Scope__enum ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef const value_type& const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::sidl::Scope > > iterator;
    typedef const_array_iter< array_traits< ::sidl::Scope > > const_iterator;
  };

  // array specialization
  template<>
  class array< ::sidl::Scope >: public enum_array< array_traits< ::sidl::Scope 
    > > {
  public:
    typedef enum_array< array_traits< ::sidl::Scope > > Base;
    typedef array_traits< ::sidl::Scope >::cxx_array_t          cxx_array_t;
    typedef array_traits< ::sidl::Scope >::cxx_item_t           cxx_item_t;
    typedef array_traits< ::sidl::Scope >::ior_array_t          ior_array_t;
    typedef array_traits< ::sidl::Scope >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::sidl::Scope >::ior_item_t           ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct sidl_Scope__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::sidl::Scope > &src) {
      d_array = src.d_array;
      if (d_array) addRef();
    }

    /**
     * assignment
     */
    array< ::sidl::Scope >&
    operator =( const array< ::sidl::Scope > &rhs) {
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
