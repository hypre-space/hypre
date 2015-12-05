/*
 * File:          SIDL_BaseClass_IOR.h
 * Symbol:        SIDL.BaseClass-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.4
 * Release:       $Name: V1-9-0b $
 * Revision:      @(#) $Id: SIDL_BaseClass_IOR.h,v 1.4 2003/04/07 21:44:24 painter Exp $
 * Description:   Intermediate Object Representation for SIDL.BaseClass
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

#ifndef included_SIDL_BaseClass_IOR_h
#define included_SIDL_BaseClass_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_IOR_h
#include "SIDL_BaseInterface_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "SIDL.BaseClass" (version 0.8.2)
 * 
 * Every class implicitly inherits from <code>BaseClass</code>.  This
 * class implements the methods in <code>BaseInterface</code>.
 */

struct SIDL_BaseClass__array;
struct SIDL_BaseClass__object;

extern struct SIDL_BaseClass__object*
SIDL_BaseClass__new(void);

extern struct SIDL_BaseClass__object*
SIDL_BaseClass__remote(const char *url);

extern void SIDL_BaseClass__init(
  struct SIDL_BaseClass__object* self);
extern void SIDL_BaseClass__fini(
  struct SIDL_BaseClass__object* self);
extern void SIDL_BaseClass__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct SIDL_BaseClass__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct SIDL_BaseClass__object* self,
    const char* name);
  void (*f__delete)(
    struct SIDL_BaseClass__object* self);
  void (*f__ctor)(
    struct SIDL_BaseClass__object* self);
  void (*f__dtor)(
    struct SIDL_BaseClass__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    struct SIDL_BaseClass__object* self);
  void (*f_deleteRef)(
    struct SIDL_BaseClass__object* self);
  SIDL_bool (*f_isSame)(
    struct SIDL_BaseClass__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct SIDL_BaseClass__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct SIDL_BaseClass__object* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct SIDL_BaseClass__object* self);
  /* Methods introduced in SIDL.BaseClass-v0.8.2 */
};

/*
 * Define the class object structure.
 */

struct SIDL_BaseClass__object {
  struct SIDL_BaseInterface__object d_sidl_baseinterface;
  struct SIDL_BaseClass__epv*       d_epv;
  void*                             d_data;
};

struct SIDL_BaseClass__external {
  struct SIDL_BaseClass__object*
  (*createObject)(void);

  struct SIDL_BaseClass__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct SIDL_BaseClass__external*
SIDL_BaseClass__externals(void);

#ifdef __cplusplus
}
#endif
#endif
