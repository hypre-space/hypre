/*
 * File:          sidl_SIDLException_IOR.h
 * Symbol:        sidl.SIDLException-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.SIDLException
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

#ifndef included_sidl_SIDLException_IOR_h
#define included_sidl_SIDLException_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif
#ifndef included_sidl_BaseException_IOR_h
#include "sidl_BaseException_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.SIDLException" (version 0.9.0)
 * 
 * <code>SIDLException</code> provides the basic functionality of the
 * <code>BaseException</code> interface for getting and setting error
 * messages and stack traces.
 */

struct sidl_SIDLException__array;
struct sidl_SIDLException__object;

extern struct sidl_SIDLException__object*
sidl_SIDLException__new(void);

extern struct sidl_SIDLException__object*
sidl_SIDLException__remote(const char *url);

extern void sidl_SIDLException__init(
  struct sidl_SIDLException__object* self);
extern void sidl_SIDLException__fini(
  struct sidl_SIDLException__object* self);
extern void sidl_SIDLException__IOR_version(int32_t *major, int32_t *minor);

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

struct sidl_SIDLException__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct sidl_SIDLException__object* self,
    const char* name);
  void (*f__delete)(
    struct sidl_SIDLException__object* self);
  void (*f__ctor)(
    struct sidl_SIDLException__object* self);
  void (*f__dtor)(
    struct sidl_SIDLException__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    struct sidl_SIDLException__object* self);
  void (*f_deleteRef)(
    struct sidl_SIDLException__object* self);
  sidl_bool (*f_isSame)(
    struct sidl_SIDLException__object* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    struct sidl_SIDLException__object* self,
    const char* name);
  sidl_bool (*f_isType)(
    struct sidl_SIDLException__object* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    struct sidl_SIDLException__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.0 */
  /* Methods introduced in sidl.BaseException-v0.9.0 */
  char* (*f_getNote)(
    struct sidl_SIDLException__object* self);
  void (*f_setNote)(
    struct sidl_SIDLException__object* self,
    const char* message);
  char* (*f_getTrace)(
    struct sidl_SIDLException__object* self);
  void (*f_addLine)(
    struct sidl_SIDLException__object* self,
    const char* traceline);
  void (*f_add)(
    struct sidl_SIDLException__object* self,
    const char* filename,
    int32_t lineno,
    const char* methodname);
  /* Methods introduced in sidl.SIDLException-v0.9.0 */
};

/*
 * Define the class object structure.
 */

struct sidl_SIDLException__object {
  struct sidl_BaseClass__object     d_sidl_baseclass;
  struct sidl_BaseException__object d_sidl_baseexception;
  struct sidl_SIDLException__epv*   d_epv;
  void*                             d_data;
};

struct sidl_SIDLException__external {
  struct sidl_SIDLException__object*
  (*createObject)(void);

  struct sidl_SIDLException__object*
  (*createRemote)(const char *url);

struct sidl_BaseClass__epv*(*getSuperEPV)(void);};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_SIDLException__external*
sidl_SIDLException__externals(void);

#ifdef __cplusplus
}
#endif
#endif
