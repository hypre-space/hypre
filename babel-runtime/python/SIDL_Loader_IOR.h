/*
 * File:          SIDL_Loader_IOR.h
 * Symbol:        SIDL.Loader-v0.8.1
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for SIDL.Loader
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
 * babel-version = 0.8.0
 */

#ifndef included_SIDL_Loader_IOR_h
#define included_SIDL_Loader_IOR_h

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
 * Symbol "SIDL.Loader" (version 0.8.1)
 * 
 * Class <code>Loader</code> manages dyanamic loading and symbol name
 * resolution for the SIDL runtime system.  The <code>Loader</code> class
 * manages a library search path and keeps a record of all libraries
 * loaded through this interface, including the initial "global" symbols
 * in the main program.  Unless explicitly set, the search path is taken
 * from the environment variable SIDL_DLL_PATH, which is a semi-colon
 * separated sequence of URIs as described in class <code>DLL</code>.
 */

struct SIDL_Loader__array;
struct SIDL_Loader__object;
struct SIDL_Loader__sepv;

extern struct SIDL_Loader__object*
SIDL_Loader__new(void);

extern struct SIDL_Loader__object*
SIDL_Loader__remote(const char *url);

extern struct SIDL_Loader__sepv*
SIDL_Loader__statics(void);

extern void SIDL_Loader__init(
  struct SIDL_Loader__object* self);
extern void SIDL_Loader__fini(
  struct SIDL_Loader__object* self);
extern void SIDL_Loader__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;
struct SIDL_DLL__array;
struct SIDL_DLL__object;

/*
 * Declare the static entry point vector.
 */

struct SIDL_Loader__sepv {
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  /* Methods introduced in SIDL.Loader-v0.8.1 */
  void (*f_setSearchPath)(
    const char* path_name);
  char* (*f_getSearchPath)(
    void);
  void (*f_addSearchPath)(
    const char* path_fragment);
  SIDL_bool (*f_loadLibrary)(
    const char* uri);
  void (*f_addDLL)(
    struct SIDL_DLL__object* dll);
  void (*f_unloadLibraries)(
    void);
  void* (*f_lookupSymbol)(
    const char* linker_name);
  struct SIDL_BaseClass__object* (*f_createClass)(
    const char* sidl_name);
};

/*
 * Declare the method entry point vector.
 */

struct SIDL_Loader__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct SIDL_Loader__object* self,
    const char* name);
  void (*f__delete)(
    struct SIDL_Loader__object* self);
  void (*f__ctor)(
    struct SIDL_Loader__object* self);
  void (*f__dtor)(
    struct SIDL_Loader__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct SIDL_Loader__object* self);
  void (*f_deleteRef)(
    struct SIDL_Loader__object* self);
  SIDL_bool (*f_isSame)(
    struct SIDL_Loader__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct SIDL_Loader__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct SIDL_Loader__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct SIDL_Loader__object* self);
  /* Methods introduced in SIDL.Loader-v0.8.1 */
};

/*
 * Define the class object structure.
 */

struct SIDL_Loader__object {
  struct SIDL_BaseClass__object d_sidl_baseclass;
  struct SIDL_Loader__epv*      d_epv;
  void*                         d_data;
};

struct SIDL_Loader__external {
  struct SIDL_Loader__object*
  (*createObject)(void);

  struct SIDL_Loader__object*
  (*createRemote)(const char *url);

  struct SIDL_Loader__sepv*
  (*getStaticEPV)(void);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct SIDL_Loader__external*
SIDL_Loader__externals(void);

#ifdef __cplusplus
}
#endif
#endif
