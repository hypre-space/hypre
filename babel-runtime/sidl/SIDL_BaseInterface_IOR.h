/*
 * File:          SIDL_BaseInterface_IOR.h
 * Symbol:        SIDL.BaseInterface-v0.7.5
 * Symbol Type:   interface
 * Babel Version: 0.7.5
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for SIDL.BaseInterface
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.5
 */

#ifndef included_SIDL_BaseInterface_IOR_h
#define included_SIDL_BaseInterface_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "SIDL.BaseInterface" (version 0.7.5)
 * 
 * Every interface in <code>SIDL</code> implicitly inherits
 * from <code>BaseInterface</code>, and it is implemented
 * by <code>BaseClass</code> below.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

extern struct SIDL_BaseInterface__object*
SIDL_BaseInterface__remote(const char *url);

/*
 * Declare the method entry point vector.
 */

struct SIDL_BaseInterface__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in SIDL.BaseInterface-v0.7.5 */
  void (*f_addReference)(
    void* self);
  void (*f_deleteReference)(
    void* self);
  SIDL_bool (*f_isSame)(
    void* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    void* self,
    const char* name);
  SIDL_bool (*f_isInstanceOf)(
    void* self,
    const char* name);
};

/*
 * Define the interface object structure.
 */

struct SIDL_BaseInterface__object {
  struct SIDL_BaseInterface__epv* d_epv;
  void*                           d_object;
};

#ifdef __cplusplus
}
#endif
#endif
