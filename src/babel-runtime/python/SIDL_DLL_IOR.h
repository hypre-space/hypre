/*
 * File:          SIDL_DLL_IOR.h
 * Symbol:        SIDL.DLL-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.4
 * Release:       $Name: V1-9-0b $
 * Revision:      @(#) $Id: SIDL_DLL_IOR.h,v 1.4 2003/04/07 21:44:24 painter Exp $
 * Description:   Intermediate Object Representation for SIDL.DLL
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
 * babel-version = 0.8.4
 */

#ifndef included_SIDL_DLL_IOR_h
#define included_SIDL_DLL_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "SIDL.DLL" (version 0.8.2)
 * 
 * The <code>DLL</code> class encapsulates access to a single
 * dynamically linked library.  DLLs are loaded at run-time using
 * the <code>loadLibrary</code> method and later unloaded using
 * <code>unloadLibrary</code>.  Symbols in a loaded library are
 * resolved to an opaque pointer by method <code>lookupSymbol</code>.
 * Class instances are created by <code>createClass</code>.
 */

struct SIDL_DLL__array;
struct SIDL_DLL__object;

extern struct SIDL_DLL__object*
SIDL_DLL__new(void);

extern struct SIDL_DLL__object*
SIDL_DLL__remote(const char *url);

extern void SIDL_DLL__init(
  struct SIDL_DLL__object* self);
extern void SIDL_DLL__fini(
  struct SIDL_DLL__object* self);
extern void SIDL_DLL__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct SIDL_DLL__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct SIDL_DLL__object* self,
    const char* name);
  void (*f__delete)(
    struct SIDL_DLL__object* self);
  void (*f__ctor)(
    struct SIDL_DLL__object* self);
  void (*f__dtor)(
    struct SIDL_DLL__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    struct SIDL_DLL__object* self);
  void (*f_deleteRef)(
    struct SIDL_DLL__object* self);
  SIDL_bool (*f_isSame)(
    struct SIDL_DLL__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct SIDL_DLL__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct SIDL_DLL__object* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct SIDL_DLL__object* self);
  /* Methods introduced in SIDL.BaseClass-v0.8.2 */
  /* Methods introduced in SIDL.DLL-v0.8.2 */
  SIDL_bool (*f_loadLibrary)(
    struct SIDL_DLL__object* self,
    const char* uri);
  char* (*f_getName)(
    struct SIDL_DLL__object* self);
  void (*f_unloadLibrary)(
    struct SIDL_DLL__object* self);
  void* (*f_lookupSymbol)(
    struct SIDL_DLL__object* self,
    const char* linker_name);
  struct SIDL_BaseClass__object* (*f_createClass)(
    struct SIDL_DLL__object* self,
    const char* sidl_name);
};

/*
 * Define the class object structure.
 */

struct SIDL_DLL__object {
  struct SIDL_BaseClass__object d_sidl_baseclass;
  struct SIDL_DLL__epv*         d_epv;
  void*                         d_data;
};

struct SIDL_DLL__external {
  struct SIDL_DLL__object*
  (*createObject)(void);

  struct SIDL_DLL__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct SIDL_DLL__external*
SIDL_DLL__externals(void);

#ifdef __cplusplus
}
#endif
#endif
