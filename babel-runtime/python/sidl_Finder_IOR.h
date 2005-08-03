/*
 * File:          sidl_Finder_IOR.h
 * Symbol:        sidl.Finder-v0.9.3
 * Symbol Type:   interface
 * Babel Version: 0.10.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.Finder
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
 * babel-version = 0.10.8
 */

#ifndef included_sidl_Finder_IOR_h
#define included_sidl_Finder_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
 * Symbol "sidl.Finder" (version 0.9.3)
 * 
 * Interface <code>Finder</code> is an interface for classes that resolve 
 * dynamic libraries.  
 * Class <code>Loader</code> takes one of these interfaces through the 
 * method <code>setFinder</code>.  If NULL is passed to setFinder, the 
 * class <code>DefaultFinder</code> is used.
 */

struct sidl_Finder__array;
struct sidl_Finder__object;

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

struct sidl_Finder__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ void* self);
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ void* self);
  void (*f_deleteRef)(
    /* in */ void* self);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ void* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self);
  /* Methods introduced in sidl.Finder-v0.9.3 */
  struct sidl_DLL__object* (*f_findLibrary)(
    /* in */ void* self,
    /* in */ const char* sidl_name,
    /* in */ const char* target,
    /* in */ enum sidl_Scope__enum lScope,
    /* in */ enum sidl_Resolve__enum lResolve);
  void (*f_setSearchPath)(
    /* in */ void* self,
    /* in */ const char* path_name);
  char* (*f_getSearchPath)(
    /* in */ void* self);
  void (*f_addSearchPath)(
    /* in */ void* self,
    /* in */ const char* path_fragment);
};

/*
 * Define the interface object structure.
 */

struct sidl_Finder__object {
  struct sidl_Finder__epv* d_epv;
  void*                    d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif
#ifndef included_sidl_Finder_IOR_h
#include "sidl_Finder_IOR.h"
#endif

/*
 * Symbol "sidl._Finder" (version 1.0)
 */

struct sidl__Finder__array;
struct sidl__Finder__object;

/*
 * Declare the method entry point vector.
 */

struct sidl__Finder__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidl__Finder__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidl__Finder__object* self);
  void (*f__exec)(
    /* in */ struct sidl__Finder__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidl__Finder__object* self);
  void (*f__ctor)(
    /* in */ struct sidl__Finder__object* self);
  void (*f__dtor)(
    /* in */ struct sidl__Finder__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidl__Finder__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidl__Finder__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl__Finder__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidl__Finder__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidl__Finder__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl__Finder__object* self);
  /* Methods introduced in sidl.Finder-v0.9.3 */
  struct sidl_DLL__object* (*f_findLibrary)(
    /* in */ struct sidl__Finder__object* self,
    /* in */ const char* sidl_name,
    /* in */ const char* target,
    /* in */ enum sidl_Scope__enum lScope,
    /* in */ enum sidl_Resolve__enum lResolve);
  void (*f_setSearchPath)(
    /* in */ struct sidl__Finder__object* self,
    /* in */ const char* path_name);
  char* (*f_getSearchPath)(
    /* in */ struct sidl__Finder__object* self);
  void (*f_addSearchPath)(
    /* in */ struct sidl__Finder__object* self,
    /* in */ const char* path_fragment);
  /* Methods introduced in sidl._Finder-v1.0 */
};

/*
 * Define the class object structure.
 */

struct sidl__Finder__object {
  struct sidl_BaseInterface__object d_sidl_baseinterface;
  struct sidl_Finder__object        d_sidl_finder;
  struct sidl__Finder__epv*         d_epv;
  void*                             d_data;
};


#ifdef __cplusplus
}
#endif
#endif
