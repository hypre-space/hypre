/*
 * File:          sidl_DLL_IOR.h
 * Symbol:        sidl.DLL-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.DLL
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
 * babel-version = 0.10.10
 */

#ifndef included_sidl_DLL_IOR_h
#define included_sidl_DLL_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.DLL" (version 0.9.3)
 * 
 * The <code>DLL</code> class encapsulates access to a single
 * dynamically linked library.  DLLs are loaded at run-time using
 * the <code>loadLibrary</code> method and later unloaded using
 * <code>unloadLibrary</code>.  Symbols in a loaded library are
 * resolved to an opaque pointer by method <code>lookupSymbol</code>.
 * Class instances are created by <code>createClass</code>.
 */

struct sidl_DLL__array;
struct sidl_DLL__object;

extern struct sidl_DLL__object*
sidl_DLL__new(void);

extern void sidl_DLL__init(
  struct sidl_DLL__object* self);
extern void sidl_DLL__fini(
  struct sidl_DLL__object* self);
extern void sidl_DLL__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_DLL__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidl_DLL__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidl_DLL__object* self);
  void (*f__exec)(
    /* in */ struct sidl_DLL__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidl_DLL__object* self);
  void (*f__ctor)(
    /* in */ struct sidl_DLL__object* self);
  void (*f__dtor)(
    /* in */ struct sidl_DLL__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidl_DLL__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidl_DLL__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_DLL__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidl_DLL__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_DLL__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_DLL__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.DLL-v0.9.3 */
  sidl_bool (*f_loadLibrary)(
    /* in */ struct sidl_DLL__object* self,
    /* in */ const char* uri,
    /* in */ sidl_bool loadGlobally,
    /* in */ sidl_bool loadLazy);
  char* (*f_getName)(
    /* in */ struct sidl_DLL__object* self);
  void (*f_unloadLibrary)(
    /* in */ struct sidl_DLL__object* self);
  void* (*f_lookupSymbol)(
    /* in */ struct sidl_DLL__object* self,
    /* in */ const char* linker_name);
  struct sidl_BaseClass__object* (*f_createClass)(
    /* in */ struct sidl_DLL__object* self,
    /* in */ const char* sidl_name);
};

/*
 * Define the class object structure.
 */

struct sidl_DLL__object {
  struct sidl_BaseClass__object d_sidl_baseclass;
  struct sidl_DLL__epv*         d_epv;
  void*                         d_data;
};

struct sidl_DLL__external {
  struct sidl_DLL__object*
  (*createObject)(void);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_DLL__external*
sidl_DLL__externals(void);

struct sidl_DLL__object* skel_sidl_DLL_fconnect_sidl_DLL(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DLL_fgetURL_sidl_DLL(struct sidl_DLL__object* obj); 

struct sidl_ClassInfo__object* skel_sidl_DLL_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DLL_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidl_DLL_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DLL_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* skel_sidl_DLL_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DLL_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
