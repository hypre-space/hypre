/*
 * File:          sidl_BaseException_IOR.h
 * Symbol:        sidl.BaseException-v0.9.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.BaseException
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
 * babel-version = 0.9.8
 */

#ifndef included_sidl_BaseException_IOR_h
#define included_sidl_BaseException_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.BaseException" (version 0.9.0)
 * 
 * Every exception implements <code>BaseException</code>. This interface
 * declares the basic functionality to get and set error messages and stack
 * traces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;

extern struct sidl_BaseException__object*
sidl_BaseException__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_BaseException__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  sidl_bool (*f_isSame)(
    void* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  sidl_bool (*f_isType)(
    void* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    void* self);
  /* Methods introduced in sidl.BaseException-v0.9.0 */
  char* (*f_getNote)(
    void* self);
  void (*f_setNote)(
    void* self,
    const char* message);
  char* (*f_getTrace)(
    void* self);
  void (*f_addLine)(
    void* self,
    const char* traceline);
  void (*f_add)(
    void* self,
    const char* filename,
    int32_t lineno,
    const char* methodname);
};

/*
 * Define the interface object structure.
 */

struct sidl_BaseException__object {
  struct sidl_BaseException__epv* d_epv;
  void*                           d_object;
};

#ifdef __cplusplus
}
#endif
#endif
