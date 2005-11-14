/*
 * File:          sidl_DFinder_IOR.h
 * Symbol:        sidl.DFinder-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.DFinder
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
 * babel-version = 0.10.12
 */

#ifndef included_sidl_DFinder_IOR_h
#define included_sidl_DFinder_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif
#ifndef included_sidl_Finder_IOR_h
#include "sidl_Finder_IOR.h"
#endif
#ifndef included_sidl_Resolve_IOR_h
#include "sidl_Resolve_IOR.h"
#endif
#ifndef included_sidl_Scope_IOR_h
#include "sidl_Scope_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.DFinder" (version 0.9.3)
 * 
 *  This class is the Default Finder.  If no Finder is set in class Loader,
 *  this finder is used.  It uses SCL files from the filesystem to
 *  resolve dynamic libraries.
 * 
 * The initial search path is taken from the SIDL_DLL_PATH
 * environment variable.
 */

struct sidl_DFinder__array;
struct sidl_DFinder__object;

extern struct sidl_DFinder__object*
sidl_DFinder__new(void);

extern void sidl_DFinder__init(
  struct sidl_DFinder__object* self);
extern void sidl_DFinder__fini(
  struct sidl_DFinder__object* self);
extern void sidl_DFinder__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_DLL__array;
struct sidl_DLL__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_DFinder__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidl_DFinder__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidl_DFinder__object* self);
  void (*f__exec)(
    /* in */ struct sidl_DFinder__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidl_DFinder__object* self);
  void (*f__ctor)(
    /* in */ struct sidl_DFinder__object* self);
  void (*f__dtor)(
    /* in */ struct sidl_DFinder__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidl_DFinder__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidl_DFinder__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_DFinder__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidl_DFinder__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_DFinder__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_DFinder__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.Finder-v0.9.3 */
  struct sidl_DLL__object* (*f_findLibrary)(
    /* in */ struct sidl_DFinder__object* self,
    /* in */ const char* sidl_name,
    /* in */ const char* target,
    /* in */ enum sidl_Scope__enum lScope,
    /* in */ enum sidl_Resolve__enum lResolve);
  void (*f_setSearchPath)(
    /* in */ struct sidl_DFinder__object* self,
    /* in */ const char* path_name);
  char* (*f_getSearchPath)(
    /* in */ struct sidl_DFinder__object* self);
  void (*f_addSearchPath)(
    /* in */ struct sidl_DFinder__object* self,
    /* in */ const char* path_fragment);
  /* Methods introduced in sidl.DFinder-v0.9.3 */
};

/*
 * Define the class object structure.
 */

struct sidl_DFinder__object {
  struct sidl_BaseClass__object d_sidl_baseclass;
  struct sidl_Finder__object    d_sidl_finder;
  struct sidl_DFinder__epv*     d_epv;
  void*                         d_data;
};

struct sidl_DFinder__external {
  struct sidl_DFinder__object*
  (*createObject)(void);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_DFinder__external*
sidl_DFinder__externals(void);

struct sidl_DFinder__object* skel_sidl_DFinder_fconnect_sidl_DFinder(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DFinder_fgetURL_sidl_DFinder(struct sidl_DFinder__object* obj); 

struct sidl_Finder__object* skel_sidl_DFinder_fconnect_sidl_Finder(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DFinder_fgetURL_sidl_Finder(struct sidl_Finder__object* obj); 

struct sidl_ClassInfo__object* skel_sidl_DFinder_fconnect_sidl_ClassInfo(char* 
  url, struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DFinder_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* 
  obj); 

struct sidl_DLL__object* skel_sidl_DFinder_fconnect_sidl_DLL(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DFinder_fgetURL_sidl_DLL(struct sidl_DLL__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidl_DFinder_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DFinder_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* skel_sidl_DFinder_fconnect_sidl_BaseClass(char* 
  url, struct sidl_BaseInterface__object **_ex);
char* skel_sidl_DFinder_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* 
  obj); 

#ifdef __cplusplus
}
#endif
#endif
