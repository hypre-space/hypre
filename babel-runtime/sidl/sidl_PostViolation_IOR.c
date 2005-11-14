/*
 * File:          sidl_PostViolation_IOR.c
 * Symbol:        sidl.PostViolation-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.PostViolation
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
#include "sidl_PostViolation_IOR.h"
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
static struct sidl_recursive_mutex_t sidl_PostViolation__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_PostViolation__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_PostViolation__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_PostViolation__mutex )==EDEADLOCK) */
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

static struct sidl_PostViolation__epv s_new_epv__sidl_postviolation;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseException__epv  s_new_epv__sidl_baseexception;
static struct sidl_BaseException__epv* s_old_epv__sidl_baseexception;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

static struct sidl_SIDLException__epv  s_new_epv__sidl_sidlexception;
static struct sidl_SIDLException__epv* s_old_epv__sidl_sidlexception;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidl_PostViolation__set_epv(
  struct sidl_PostViolation__epv* epv);
extern void sidl_PostViolation__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidl_PostViolation_addRef__exec(
        struct sidl_PostViolation__object* self,
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
sidl_PostViolation_deleteRef__exec(
        struct sidl_PostViolation__object* self,
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
sidl_PostViolation_isSame__exec(
        struct sidl_PostViolation__object* self,
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
sidl_PostViolation_queryInt__exec(
        struct sidl_PostViolation__object* self,
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
sidl_PostViolation_isType__exec(
        struct sidl_PostViolation__object* self,
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
sidl_PostViolation_getClassInfo__exec(
        struct sidl_PostViolation__object* self,
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

static void
sidl_PostViolation_getNote__exec(
        struct sidl_PostViolation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* _retval = 0;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getNote)(
    self);

  /* pack return value */
  sidl_io_Serializer_packString( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidl_PostViolation_setNote__exec(
        struct sidl_PostViolation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* message= NULL;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "message", &message, _ex2);

  /* make the call */
  (self->d_epv->f_setNote)(
    self,
    message);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidl_PostViolation_getTrace__exec(
        struct sidl_PostViolation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* _retval = 0;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getTrace)(
    self);

  /* pack return value */
  sidl_io_Serializer_packString( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidl_PostViolation_addLine__exec(
        struct sidl_PostViolation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* traceline= NULL;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "traceline", &traceline, _ex2);

  /* make the call */
  (self->d_epv->f_addLine)(
    self,
    traceline);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidl_PostViolation_add__exec(
        struct sidl_PostViolation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* filename= NULL;
  int32_t lineno = 0;
  char* methodname= NULL;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "filename", &filename, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "lineno", &lineno, _ex2);
  sidl_io_Deserializer_unpackString( inArgs, "methodname", &methodname, _ex2);

  /* make the call */
  (self->d_epv->f_add)(
    self,
    filename,
    lineno,
    methodname);

  /* pack return value */
  /* pack out and inout argments */

}

static void ior_sidl_PostViolation__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidl_PostViolation__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_sidl_PostViolation__cast(
  struct sidl_PostViolation__object* self,
  const char* name)
{
  void* cast = NULL;

  struct sidl_PostViolation__object* s0 = self;
  struct sidl_SIDLException__object* s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*     s2 = &s1->d_sidl_baseclass;

  if (!strcmp(name, "sidl.PostViolation")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidl.SIDLException")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseException")) {
    cast = (void*) &s1->d_sidl_baseexception;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s2;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s2->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_sidl_PostViolation__delete(
  struct sidl_PostViolation__object* self)
{
  sidl_PostViolation__fini(self);
  memset((void*)self, 0, sizeof(struct sidl_PostViolation__object));
  free((void*) self);
}

static char*
ior_sidl_PostViolation__getURL(
    struct sidl_PostViolation__object* self) {
  /* TODO: Make this work for local object! */
  return NULL;
}
struct sidl_PostViolation__method {
  const char *d_name;
  void (*d_func)(struct sidl_PostViolation__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_sidl_PostViolation__exec(
    struct sidl_PostViolation__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct sidl_PostViolation__method  s_methods[] = {
    { "add", sidl_PostViolation_add__exec },
    { "addLine", sidl_PostViolation_addLine__exec },
    { "addRef", sidl_PostViolation_addRef__exec },
    { "deleteRef", sidl_PostViolation_deleteRef__exec },
    { "getClassInfo", sidl_PostViolation_getClassInfo__exec },
    { "getNote", sidl_PostViolation_getNote__exec },
    { "getTrace", sidl_PostViolation_getTrace__exec },
    { "isSame", sidl_PostViolation_isSame__exec },
    { "isType", sidl_PostViolation_isType__exec },
    { "queryInt", sidl_PostViolation_queryInt__exec },
    { "setNote", sidl_PostViolation_setNote__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidl_PostViolation__method);
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

static void sidl_PostViolation__init_epv(
  struct sidl_PostViolation__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidl_PostViolation__object* s0 = self;
  struct sidl_SIDLException__object* s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*     s2 = &s1->d_sidl_baseclass;

  struct sidl_PostViolation__epv*  epv  = &s_new_epv__sidl_postviolation;
  struct sidl_BaseClass__epv*      e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseException__epv*  e1   = &s_new_epv__sidl_baseexception;
  struct sidl_BaseInterface__epv*  e2   = &s_new_epv__sidl_baseinterface;
  struct sidl_SIDLException__epv*  e3   = &s_new_epv__sidl_sidlexception;

  s_old_epv__sidl_baseinterface = s2->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s2->d_epv;

  s_old_epv__sidl_baseexception = s1->d_sidl_baseexception.d_epv;
  s_old_epv__sidl_sidlexception = s1->d_epv;

  epv->f__cast                    = ior_sidl_PostViolation__cast;
  epv->f__delete                  = ior_sidl_PostViolation__delete;
  epv->f__exec                    = ior_sidl_PostViolation__exec;
  epv->f__getURL                  = ior_sidl_PostViolation__getURL;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    sidl_PostViolation__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidl_PostViolation__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidl_PostViolation__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct sidl_PostViolation__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidl_PostViolation__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_PostViolation__object*)) s1->d_epv->f_getClassInfo;
  epv->f_getNote                  = (char* (*)(struct 
    sidl_PostViolation__object*)) s1->d_epv->f_getNote;
  epv->f_setNote                  = (void (*)(struct 
    sidl_PostViolation__object*,const char*)) s1->d_epv->f_setNote;
  epv->f_getTrace                 = (char* (*)(struct 
    sidl_PostViolation__object*)) s1->d_epv->f_getTrace;
  epv->f_addLine                  = (void (*)(struct 
    sidl_PostViolation__object*,const char*)) s1->d_epv->f_addLine;
  epv->f_add                      = (void (*)(struct 
    sidl_PostViolation__object*,const char*,int32_t,
    const char*)) s1->d_epv->f_add;

  sidl_PostViolation__set_epv(epv);

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
  e1->f_getNote             = (char* (*)(void*)) epv->f_getNote;
  e1->f_setNote             = (void (*)(void*,const char*)) epv->f_setNote;
  e1->f_getTrace            = (char* (*)(void*)) epv->f_getTrace;
  e1->f_addLine             = (void (*)(void*,const char*)) epv->f_addLine;
  e1->f_add                 = (void (*)(void*,const char*,int32_t,
    const char*)) epv->f_add;

  e2->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete             = (void (*)(void*)) epv->f__delete;
  e2->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  e3->f__cast               = (void* (*)(struct sidl_SIDLException__object*,
    const char*)) epv->f__cast;
  e3->f__delete             = (void (*)(struct sidl_SIDLException__object*)) 
    epv->f__delete;
  e3->f__exec               = (void (*)(struct sidl_SIDLException__object*,
    const char*,struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef              = (void (*)(struct sidl_SIDLException__object*)) 
    epv->f_addRef;
  e3->f_deleteRef           = (void (*)(struct sidl_SIDLException__object*)) 
    epv->f_deleteRef;
  e3->f_isSame              = (sidl_bool (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt            = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_SIDLException__object*,const char*)) epv->f_queryInt;
  e3->f_isType              = (sidl_bool (*)(struct sidl_SIDLException__object*,
    const char*)) epv->f_isType;
  e3->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_SIDLException__object*)) epv->f_getClassInfo;
  e3->f_getNote             = (char* (*)(struct sidl_SIDLException__object*)) 
    epv->f_getNote;
  e3->f_setNote             = (void (*)(struct sidl_SIDLException__object*,
    const char*)) epv->f_setNote;
  e3->f_getTrace            = (char* (*)(struct sidl_SIDLException__object*)) 
    epv->f_getTrace;
  e3->f_addLine             = (void (*)(struct sidl_SIDLException__object*,
    const char*)) epv->f_addLine;
  e3->f_add                 = (void (*)(struct sidl_SIDLException__object*,
    const char*,int32_t,const char*)) epv->f_add;

  s_method_initialized = 1;
  ior_sidl_PostViolation__ensure_load_called();
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_SIDLException__epv* sidl_PostViolation__super(void) {
  return s_old_epv__sidl_sidlexception;
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
      sidl_ClassInfoI_setName(impl, "sidl.PostViolation");
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
initMetadata(struct sidl_PostViolation__object* self)
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

struct sidl_PostViolation__object*
sidl_PostViolation__new(void)
{
  struct sidl_PostViolation__object* self =
    (struct sidl_PostViolation__object*) malloc(
      sizeof(struct sidl_PostViolation__object));
  sidl_PostViolation__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidl_PostViolation__init(
  struct sidl_PostViolation__object* self)
{
  struct sidl_PostViolation__object* s0 = self;
  struct sidl_SIDLException__object* s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*     s2 = &s1->d_sidl_baseclass;

  sidl_SIDLException__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidl_PostViolation__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s2->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s2->d_epv                      = &s_new_epv__sidl_baseclass;

  s1->d_sidl_baseexception.d_epv = &s_new_epv__sidl_baseexception;
  s1->d_epv                      = &s_new_epv__sidl_sidlexception;

  s0->d_epv    = &s_new_epv__sidl_postviolation;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void sidl_PostViolation__fini(
  struct sidl_PostViolation__object* self)
{
  struct sidl_PostViolation__object* s0 = self;
  struct sidl_SIDLException__object* s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*     s2 = &s1->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s2->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s2->d_epv                      = s_old_epv__sidl_baseclass;

  s1->d_sidl_baseexception.d_epv = s_old_epv__sidl_baseexception;
  s1->d_epv                      = s_old_epv__sidl_sidlexception;

  sidl_SIDLException__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidl_PostViolation__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidl_PostViolation__external
s_externalEntryPoints = {
  sidl_PostViolation__new,
  sidl_PostViolation__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_PostViolation__external*
sidl_PostViolation__externals(void)
{
  return &s_externalEntryPoints;
}

