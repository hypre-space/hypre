/*
 * File:          sidl_SIDLException_IOR.c
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

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "sidl_SIDLException_IOR.h"
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

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 8;
/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct sidl_SIDLException__epv s_new__sidl_sidlexception;
static struct sidl_SIDLException__epv s_rem__sidl_sidlexception;

static struct sidl_BaseClass__epv  s_new__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old__sidl_baseclass;
static struct sidl_BaseClass__epv  s_rem__sidl_baseclass;

static struct sidl_BaseException__epv s_new__sidl_baseexception;
static struct sidl_BaseException__epv s_rem__sidl_baseexception;

static struct sidl_BaseInterface__epv  s_new__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old__sidl_baseinterface;
static struct sidl_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidl_SIDLException__set_epv(
  struct sidl_SIDLException__epv* epv);
#ifdef __cplusplus
}
#endif

/*
 * CAST: dynamic type casting support.
 */

static void* ior_sidl_SIDLException__cast(
  struct sidl_SIDLException__object* self,
  const char* name)
{
  void* cast = NULL;

  struct sidl_SIDLException__object* s0 = self;
  struct sidl_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "sidl.SIDLException")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidl.BaseException")) {
    cast = (void*) &s0->d_sidl_baseexception;
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

static void ior_sidl_SIDLException__delete(
  struct sidl_SIDLException__object* self)
{
  sidl_SIDLException__fini(self);
  memset((void*)self, 0, sizeof(struct sidl_SIDLException__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void sidl_SIDLException__init_epv(
  struct sidl_SIDLException__object* self)
{
  struct sidl_SIDLException__object* s0 = self;
  struct sidl_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  struct sidl_SIDLException__epv* epv = &s_new__sidl_sidlexception;
  struct sidl_BaseClass__epv*     e0  = &s_new__sidl_baseclass;
  struct sidl_BaseException__epv* e1  = &s_new__sidl_baseexception;
  struct sidl_BaseInterface__epv* e2  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast        = ior_sidl_SIDLException__cast;
  epv->f__delete      = ior_sidl_SIDLException__delete;
  epv->f__ctor        = NULL;
  epv->f__dtor        = NULL;
  epv->f_addRef       = (void (*)(struct sidl_SIDLException__object*)) 
    s1->d_epv->f_addRef;
  epv->f_deleteRef    = (void (*)(struct sidl_SIDLException__object*)) 
    s1->d_epv->f_deleteRef;
  epv->f_isSame       = (sidl_bool (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_SIDLException__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType       = (sidl_bool (*)(struct sidl_SIDLException__object*,
    const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_SIDLException__object*)) s1->d_epv->f_getClassInfo;
  epv->f_getNote      = NULL;
  epv->f_setNote      = NULL;
  epv->f_getTrace     = NULL;
  epv->f_addLine      = NULL;
  epv->f_add          = NULL;

  sidl_SIDLException__set_epv(epv);

  e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_getNote      = (char* (*)(void*)) epv->f_getNote;
  e1->f_setNote      = (void (*)(void*,const char*)) epv->f_setNote;
  e1->f_getTrace     = (char* (*)(void*)) epv->f_getTrace;
  e1->f_addLine      = (void (*)(void*,const char*)) epv->f_addLine;
  e1->f_add          = (void (*)(void*,const char*,int32_t,
    const char*)) epv->f_add;

  e2->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete      = (void (*)(void*)) epv->f__delete;
  e2->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_method_initialized = 1;
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* sidl_SIDLException__super(void) {
  return s_old__sidl_baseclass;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info)
{
  if (s_classInfo_init) {
    sidl_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = sidl_ClassInfoI__create();
    s_classInfo = sidl_ClassInfo__cast(impl);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "sidl.SIDLException");
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
    }
  }
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info);
  }
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct sidl_SIDLException__object* self)
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

struct sidl_SIDLException__object*
sidl_SIDLException__new(void)
{
  struct sidl_SIDLException__object* self =
    (struct sidl_SIDLException__object*) malloc(
      sizeof(struct sidl_SIDLException__object));
  sidl_SIDLException__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidl_SIDLException__init(
  struct sidl_SIDLException__object* self)
{
  struct sidl_SIDLException__object* s0 = self;
  struct sidl_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  if (!s_method_initialized) {
    sidl_SIDLException__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_sidl_baseexception.d_epv = &s_new__sidl_baseexception;
  s0->d_epv                      = &s_new__sidl_sidlexception;

  s0->d_sidl_baseexception.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void sidl_SIDLException__fini(
  struct sidl_SIDLException__object* self)
{
  struct sidl_SIDLException__object* s0 = self;
  struct sidl_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidl_SIDLException__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct sidl_SIDLException__external
s_externalEntryPoints = {
  sidl_SIDLException__new,
  sidl_SIDLException__remote,
  sidl_SIDLException__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_SIDLException__external*
sidl_SIDLException__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_sidl_SIDLException__cast(
  struct sidl_SIDLException__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_sidl_SIDLException__delete(
  struct sidl_SIDLException__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_sidl_SIDLException_addRef(
  struct sidl_SIDLException__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_sidl_SIDLException_deleteRef(
  struct sidl_SIDLException__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_sidl_SIDLException_isSame(
  struct sidl_SIDLException__object* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_sidl_SIDLException_queryInt(
  struct sidl_SIDLException__object* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_sidl_SIDLException_isType(
  struct sidl_SIDLException__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_sidl_SIDLException_getClassInfo(
  struct sidl_SIDLException__object* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:getNote
 */

static char*
remote_sidl_SIDLException_getNote(
  struct sidl_SIDLException__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:setNote
 */

static void
remote_sidl_SIDLException_setNote(
  struct sidl_SIDLException__object* self,
  const char* message)
{
}

/*
 * REMOTE METHOD STUB:getTrace
 */

static char*
remote_sidl_SIDLException_getTrace(
  struct sidl_SIDLException__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:addLine
 */

static void
remote_sidl_SIDLException_addLine(
  struct sidl_SIDLException__object* self,
  const char* traceline)
{
}

/*
 * REMOTE METHOD STUB:add
 */

static void
remote_sidl_SIDLException_add(
  struct sidl_SIDLException__object* self,
  const char* filename,
  int32_t lineno,
  const char* methodname)
{
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void sidl_SIDLException__init_remote_epv(void)
{
  struct sidl_SIDLException__epv* epv = &s_rem__sidl_sidlexception;
  struct sidl_BaseClass__epv*     e0  = &s_rem__sidl_baseclass;
  struct sidl_BaseException__epv* e1  = &s_rem__sidl_baseexception;
  struct sidl_BaseInterface__epv* e2  = &s_rem__sidl_baseinterface;

  epv->f__cast        = remote_sidl_SIDLException__cast;
  epv->f__delete      = remote_sidl_SIDLException__delete;
  epv->f__ctor        = NULL;
  epv->f__dtor        = NULL;
  epv->f_addRef       = remote_sidl_SIDLException_addRef;
  epv->f_deleteRef    = remote_sidl_SIDLException_deleteRef;
  epv->f_isSame       = remote_sidl_SIDLException_isSame;
  epv->f_queryInt     = remote_sidl_SIDLException_queryInt;
  epv->f_isType       = remote_sidl_SIDLException_isType;
  epv->f_getClassInfo = remote_sidl_SIDLException_getClassInfo;
  epv->f_getNote      = remote_sidl_SIDLException_getNote;
  epv->f_setNote      = remote_sidl_SIDLException_setNote;
  epv->f_getTrace     = remote_sidl_SIDLException_getTrace;
  epv->f_addLine      = remote_sidl_SIDLException_addLine;
  epv->f_add          = remote_sidl_SIDLException_add;

  e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_getNote      = (char* (*)(void*)) epv->f_getNote;
  e1->f_setNote      = (void (*)(void*,const char*)) epv->f_setNote;
  e1->f_getTrace     = (char* (*)(void*)) epv->f_getTrace;
  e1->f_addLine      = (void (*)(void*,const char*)) epv->f_addLine;
  e1->f_add          = (void (*)(void*,const char*,int32_t,
    const char*)) epv->f_add;

  e2->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete      = (void (*)(void*)) epv->f__delete;
  e2->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct sidl_SIDLException__object*
sidl_SIDLException__remote(const char *url)
{
  struct sidl_SIDLException__object* self =
    (struct sidl_SIDLException__object*) malloc(
      sizeof(struct sidl_SIDLException__object));

  struct sidl_SIDLException__object* s0 = self;
  struct sidl_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    sidl_SIDLException__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_sidl_baseexception.d_epv    = &s_rem__sidl_baseexception;
  s0->d_sidl_baseexception.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__sidl_sidlexception;

  return self;
}
