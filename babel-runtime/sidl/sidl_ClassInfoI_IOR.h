/*
 * File:          sidl_ClassInfoI_IOR.h
 * Symbol:        sidl.ClassInfoI-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.ClassInfoI
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

#ifndef included_sidl_ClassInfoI_IOR_h
#define included_sidl_ClassInfoI_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif
#ifndef included_sidl_ClassInfo_IOR_h
#include "sidl_ClassInfo_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.ClassInfoI" (version 0.9.3)
 * 
 * An implementation of the <code>ClassInfo</code> interface. This provides
 * methods to set all the attributes that are read-only in the
 * <code>ClassInfo</code> interface.
 */

struct sidl_ClassInfoI__array;
struct sidl_ClassInfoI__object;

extern struct sidl_ClassInfoI__object*
sidl_ClassInfoI__new(void);

extern void sidl_ClassInfoI__init(
  struct sidl_ClassInfoI__object* self);
extern void sidl_ClassInfoI__fini(
  struct sidl_ClassInfoI__object* self);
extern void sidl_ClassInfoI__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_ClassInfoI__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidl_ClassInfoI__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidl_ClassInfoI__object* self);
  void (*f__exec)(
    /* in */ struct sidl_ClassInfoI__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidl_ClassInfoI__object* self);
  void (*f__ctor)(
    /* in */ struct sidl_ClassInfoI__object* self);
  void (*f__dtor)(
    /* in */ struct sidl_ClassInfoI__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidl_ClassInfoI__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidl_ClassInfoI__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_ClassInfoI__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidl_ClassInfoI__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_ClassInfoI__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_ClassInfoI__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.ClassInfo-v0.9.3 */
  char* (*f_getName)(
    /* in */ struct sidl_ClassInfoI__object* self);
  char* (*f_getIORVersion)(
    /* in */ struct sidl_ClassInfoI__object* self);
  /* Methods introduced in sidl.ClassInfoI-v0.9.3 */
  void (*f_setName)(
    /* in */ struct sidl_ClassInfoI__object* self,
    /* in */ const char* name);
  void (*f_setIORVersion)(
    /* in */ struct sidl_ClassInfoI__object* self,
    /* in */ int32_t major,
    /* in */ int32_t minor);
};

/*
 * Define the class object structure.
 */

struct sidl_ClassInfoI__object {
  struct sidl_BaseClass__object d_sidl_baseclass;
  struct sidl_ClassInfo__object d_sidl_classinfo;
  struct sidl_ClassInfoI__epv*  d_epv;
  void*                         d_data;
};

struct sidl_ClassInfoI__external {
  struct sidl_ClassInfoI__object*
  (*createObject)(void);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_ClassInfoI__external*
sidl_ClassInfoI__externals(void);

struct sidl_ClassInfoI__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_ClassInfoI_fgetURL_sidl_ClassInfoI(struct 
  sidl_ClassInfoI__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_ClassInfoI_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_ClassInfoI_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_ClassInfoI_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
