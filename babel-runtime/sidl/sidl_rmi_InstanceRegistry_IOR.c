/*
 * File:          sidl_rmi_InstanceRegistry_IOR.c
 * Symbol:        sidl.rmi.InstanceRegistry-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.rmi.InstanceRegistry
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

#include "sidl_rmi_InstanceHandle.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "sidl_rmi_InstanceRegistry_IOR.h"
#ifndef included_sidl_BaseClass_Impl_h
#include "sidl_BaseClass_Impl.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_ClassInfoI_h
#include "sidl_ClassInfoI.h"
#endif

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidl_rmi_InstanceRegistry__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_rmi_InstanceRegistry__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_rmi_InstanceRegistry__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_rmi_InstanceRegistry__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 9;

/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variable to make sure _load called no more than once
 */

static int s_load_called = 0;
/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_static_initialized = 0;

static struct sidl_rmi_InstanceRegistry__epv  
  s_new_epv__sidl_rmi_instanceregistry;
static struct sidl_rmi_InstanceRegistry__sepv 
  s_stc_epv__sidl_rmi_instanceregistry;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidl_rmi_InstanceRegistry__set_epv(
  struct sidl_rmi_InstanceRegistry__epv* epv);
extern void sidl_rmi_InstanceRegistry__set_sepv(
  struct sidl_rmi_InstanceRegistry__sepv* sepv);
extern void sidl_rmi_InstanceRegistry__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidl_rmi_InstanceRegistry_addRef__exec(
        struct sidl_rmi_InstanceRegistry__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_addRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidl_rmi_InstanceRegistry_deleteRef__exec(
        struct sidl_rmi_InstanceRegistry__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_deleteRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidl_rmi_InstanceRegistry_isSame__exec(
        struct sidl_rmi_InstanceRegistry__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_BaseInterface__object* iobj = 0;
  sidl_bool _retval = FALSE;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_isSame)(
    self,
    iobj);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidl_rmi_InstanceRegistry_queryInt__exec(
        struct sidl_rmi_InstanceRegistry__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_BaseInterface__object* _retval = 0;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_queryInt)(
    self,
    name);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidl_rmi_InstanceRegistry_isType__exec(
        struct sidl_rmi_InstanceRegistry__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  sidl_bool _retval = FALSE;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_isType)(
    self,
    name);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidl_rmi_InstanceRegistry_getClassInfo__exec(
        struct sidl_rmi_InstanceRegistry__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_ClassInfo__object* _retval = 0;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getClassInfo)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void ior_sidl_rmi_InstanceRegistry__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidl_rmi_InstanceRegistry__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_sidl_rmi_InstanceRegistry__cast(
  struct sidl_rmi_InstanceRegistry__object* self,
  const char* name)
{
  void* cast = NULL;

  struct sidl_rmi_InstanceRegistry__object* s0 = self;
  struct sidl_BaseClass__object*            s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "sidl.rmi.InstanceRegistry")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_sidl_rmi_InstanceRegistry__delete(
  struct sidl_rmi_InstanceRegistry__object* self)
{
  sidl_rmi_InstanceRegistry__fini(self);
  memset((void*)self, 0, sizeof(struct sidl_rmi_InstanceRegistry__object));
  free((void*) self);
}

static char*
ior_sidl_rmi_InstanceRegistry__getURL(
    struct sidl_rmi_InstanceRegistry__object* self) {
  /* TODO: Make this work for local object! */
  return NULL;
}
struct sidl_rmi_InstanceRegistry__method {
  const char *d_name;
  void (*d_func)(struct sidl_rmi_InstanceRegistry__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_sidl_rmi_InstanceRegistry__exec(
    struct sidl_rmi_InstanceRegistry__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct sidl_rmi_InstanceRegistry__method  s_methods[] = {
    { "addRef", sidl_rmi_InstanceRegistry_addRef__exec },
    { "deleteRef", sidl_rmi_InstanceRegistry_deleteRef__exec },
    { "getClassInfo", sidl_rmi_InstanceRegistry_getClassInfo__exec },
    { "isSame", sidl_rmi_InstanceRegistry_isSame__exec },
    { "isType", sidl_rmi_InstanceRegistry_isType__exec },
    { "queryInt", sidl_rmi_InstanceRegistry_queryInt__exec },
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidl_rmi_InstanceRegistry__method);
  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(self, inArgs, outArgs);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
}
/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void sidl_rmi_InstanceRegistry__init_epv(
  struct sidl_rmi_InstanceRegistry__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidl_rmi_InstanceRegistry__object* s0 = self;
  struct sidl_BaseClass__object*            s1 = &s0->d_sidl_baseclass;

  struct sidl_rmi_InstanceRegistry__epv*  epv  = 
    &s_new_epv__sidl_rmi_instanceregistry;
  struct sidl_BaseClass__epv*             e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*         e1   = &s_new_epv__sidl_baseinterface;

  s_old_epv__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s1->d_epv;

  epv->f__cast                    = ior_sidl_rmi_InstanceRegistry__cast;
  epv->f__delete                  = ior_sidl_rmi_InstanceRegistry__delete;
  epv->f__exec                    = ior_sidl_rmi_InstanceRegistry__exec;
  epv->f__getURL                  = ior_sidl_rmi_InstanceRegistry__getURL;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    sidl_rmi_InstanceRegistry__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidl_rmi_InstanceRegistry__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidl_rmi_InstanceRegistry__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct sidl_rmi_InstanceRegistry__object*,
    const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidl_rmi_InstanceRegistry__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_rmi_InstanceRegistry__object*)) s1->d_epv->f_getClassInfo;

  sidl_rmi_InstanceRegistry__set_epv(epv);

  e0->f__cast               = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete             = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e0->f__exec               = (void (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef              = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_addRef;
  e0->f_deleteRef           = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt            = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete             = (void (*)(void*)) epv->f__delete;
  e1->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_method_initialized = 1;
  ior_sidl_rmi_InstanceRegistry__ensure_load_called();
}

/*
 * SEPV: create the static entry point vector (SEPV).
 */

static void sidl_rmi_InstanceRegistry__init_sepv(void)
{
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  struct sidl_rmi_InstanceRegistry__sepv*  s = 
    &s_stc_epv__sidl_rmi_instanceregistry;

  s->f_registerInstance         = NULL;
  s->f_getInstance              = NULL;
  s->f_removeInstance           = NULL;

  sidl_rmi_InstanceRegistry__set_sepv(s);

  s_static_initialized = 1;
  ior_sidl_rmi_InstanceRegistry__ensure_load_called();
}

/*
 * STATIC: return pointer to static EPV structure.
 */

struct sidl_rmi_InstanceRegistry__sepv*
sidl_rmi_InstanceRegistry__statics(void)
{
  LOCK_STATIC_GLOBALS;
  if (!s_static_initialized) {
    sidl_rmi_InstanceRegistry__init_sepv();
  }
  UNLOCK_STATIC_GLOBALS;
  return &s_stc_epv__sidl_rmi_instanceregistry;
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* sidl_rmi_InstanceRegistry__super(void) {
  return s_old_epv__sidl_baseclass;
}

static void
cleanupClassInfo(void) {
  if (s_classInfo) {
    sidl_ClassInfo_deleteRef(s_classInfo);
  }
  s_classInfo_init = 1;
  s_classInfo = NULL;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info)
{
  LOCK_STATIC_GLOBALS;
  if (s_classInfo_init) {
    sidl_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = sidl_ClassInfoI__create();
    s_classInfo = sidl_ClassInfo__cast(impl);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "sidl.rmi.InstanceRegistry");
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
      atexit(cleanupClassInfo);
    }
  }
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info);
  }
UNLOCK_STATIC_GLOBALS;
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct sidl_rmi_InstanceRegistry__object* self)
{
  if (self) {
    struct sidl_BaseClass__data *data = 
      sidl_BaseClass__get_data(sidl_BaseClass__cast(self));
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo));
    }
  }
}

/*
 * NEW: allocate object and initialize it.
 */

struct sidl_rmi_InstanceRegistry__object*
sidl_rmi_InstanceRegistry__new(void)
{
  struct sidl_rmi_InstanceRegistry__object* self =
    (struct sidl_rmi_InstanceRegistry__object*) malloc(
      sizeof(struct sidl_rmi_InstanceRegistry__object));
  sidl_rmi_InstanceRegistry__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidl_rmi_InstanceRegistry__init(
  struct sidl_rmi_InstanceRegistry__object* self)
{
  struct sidl_rmi_InstanceRegistry__object* s0 = self;
  struct sidl_BaseClass__object*            s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidl_rmi_InstanceRegistry__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s1->d_epv                      = &s_new_epv__sidl_baseclass;

  s0->d_epv    = &s_new_epv__sidl_rmi_instanceregistry;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void sidl_rmi_InstanceRegistry__fini(
  struct sidl_rmi_InstanceRegistry__object* self)
{
  struct sidl_rmi_InstanceRegistry__object* s0 = self;
  struct sidl_BaseClass__object*            s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s1->d_epv                      = s_old_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidl_rmi_InstanceRegistry__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidl_rmi_InstanceRegistry__external
s_externalEntryPoints = {
  sidl_rmi_InstanceRegistry__new,
  sidl_rmi_InstanceRegistry__statics,
  sidl_rmi_InstanceRegistry__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_rmi_InstanceRegistry__external*
sidl_rmi_InstanceRegistry__externals(void)
{
  return &s_externalEntryPoints;
}

