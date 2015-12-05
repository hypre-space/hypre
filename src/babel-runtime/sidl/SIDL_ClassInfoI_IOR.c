/*
 * File:          SIDL_ClassInfoI_IOR.c
 * Symbol:        SIDL.ClassInfoI-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.4
 * Release:       $Name: V1-9-0b $
 * Revision:      @(#) $Id: SIDL_ClassInfoI_IOR.c,v 1.3 2003/04/07 21:44:31 painter Exp $
 * Description:   Intermediate Object Representation for SIDL.ClassInfoI
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

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "SIDL_ClassInfoI_IOR.h"
#ifndef included_SIDL_BaseClass_Impl_h
#include "SIDL_BaseClass_Impl.h"
#endif
#ifndef included_SIDL_BaseClass_h
#include "SIDL_BaseClass.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif
#ifndef included_SIDL_ClassInfoI_h
#include "SIDL_ClassInfoI.h"
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

static SIDL_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct SIDL_ClassInfoI__epv s_new__sidl_classinfoi;
static struct SIDL_ClassInfoI__epv s_rem__sidl_classinfoi;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

static struct SIDL_ClassInfo__epv s_new__sidl_classinfo;
static struct SIDL_ClassInfo__epv s_rem__sidl_classinfo;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void SIDL_ClassInfoI__set_epv(
  struct SIDL_ClassInfoI__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* ior_SIDL_ClassInfoI__cast(
  struct SIDL_ClassInfoI__object* self,
  const char* name)
{
  void* cast = NULL;

  struct SIDL_ClassInfoI__object* s0 = self;
  struct SIDL_BaseClass__object*  s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "SIDL.ClassInfoI")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "SIDL.ClassInfo")) {
    cast = (void*) &s0->d_sidl_classinfo;
  } else if (!strcmp(name, "SIDL.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "SIDL.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_SIDL_ClassInfoI__delete(
  struct SIDL_ClassInfoI__object* self)
{
  SIDL_ClassInfoI__fini(self);
  memset((void*)self, 0, sizeof(struct SIDL_ClassInfoI__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void SIDL_ClassInfoI__init_epv(
  struct SIDL_ClassInfoI__object* self)
{
  struct SIDL_ClassInfoI__object* s0 = self;
  struct SIDL_BaseClass__object*  s1 = &s0->d_sidl_baseclass;

  struct SIDL_ClassInfoI__epv*    epv = &s_new__sidl_classinfoi;
  struct SIDL_BaseClass__epv*     e0  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e1  = &s_new__sidl_baseinterface;
  struct SIDL_ClassInfo__epv*     e2  = &s_new__sidl_classinfo;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast         = ior_SIDL_ClassInfoI__cast;
  epv->f__delete       = ior_SIDL_ClassInfoI__delete;
  epv->f__ctor         = NULL;
  epv->f__dtor         = NULL;
  epv->f_addRef        = (void (*)(struct SIDL_ClassInfoI__object*)) 
    s1->d_epv->f_addRef;
  epv->f_deleteRef     = (void (*)(struct SIDL_ClassInfoI__object*)) 
    s1->d_epv->f_deleteRef;
  epv->f_isSame        = (SIDL_bool (*)(struct SIDL_ClassInfoI__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt      = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_ClassInfoI__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType        = (SIDL_bool (*)(struct SIDL_ClassInfoI__object*,
    const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo  = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_ClassInfoI__object*)) s1->d_epv->f_getClassInfo;
  epv->f_setName       = NULL;
  epv->f_setIORVersion = NULL;
  epv->f_getName       = NULL;
  epv->f_getIORVersion = NULL;

  SIDL_ClassInfoI__set_epv(epv);

  e0->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  e2->f__cast         = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete       = (void (*)(void*)) epv->f__delete;
  e2->f_addRef        = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef     = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame        = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt      = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType        = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo  = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_getName       = (char* (*)(void*)) epv->f_getName;
  e2->f_getIORVersion = (char* (*)(void*)) epv->f_getIORVersion;

  s_method_initialized = 1;
}

static void initMetadata(struct SIDL_ClassInfoI__object*);

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(SIDL_ClassInfo *info)
{
  if (s_classInfo_init) {
    SIDL_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = SIDL_ClassInfoI__create();
    s_classInfo = SIDL_ClassInfo__cast(impl);
    if (impl) {
      SIDL_ClassInfoI_setName(impl, "SIDL.ClassInfoI");
      SIDL_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
      initMetadata(impl);
    }
  }
  if (s_classInfo) {
    if (*info) {
      SIDL_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    SIDL_ClassInfo_addRef(*info);
  }
}

/*
 * initMetadata: store IOR version & class in SIDL.BaseClass's data
 */

static void
initMetadata(struct SIDL_ClassInfoI__object* self)
{
  if (self) {
    struct SIDL_BaseClass__data *data = 
      SIDL_BaseClass__get_data(SIDL_BaseClass__cast(self));
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

struct SIDL_ClassInfoI__object*
SIDL_ClassInfoI__new(void)
{
  struct SIDL_ClassInfoI__object* self =
    (struct SIDL_ClassInfoI__object*) malloc(
      sizeof(struct SIDL_ClassInfoI__object));
  SIDL_ClassInfoI__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void SIDL_ClassInfoI__init(
  struct SIDL_ClassInfoI__object* self)
{
  struct SIDL_ClassInfoI__object* s0 = self;
  struct SIDL_BaseClass__object*  s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    SIDL_ClassInfoI__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_sidl_classinfo.d_epv = &s_new__sidl_classinfo;
  s0->d_epv                  = &s_new__sidl_classinfoi;

  s0->d_sidl_classinfo.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void SIDL_ClassInfoI__fini(
  struct SIDL_ClassInfoI__object* self)
{
  struct SIDL_ClassInfoI__object* s0 = self;
  struct SIDL_BaseClass__object*  s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
SIDL_ClassInfoI__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct SIDL_ClassInfoI__external
s_externalEntryPoints = {
  SIDL_ClassInfoI__new,
  SIDL_ClassInfoI__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct SIDL_ClassInfoI__external*
SIDL_ClassInfoI__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_SIDL_ClassInfoI__cast(
  struct SIDL_ClassInfoI__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_SIDL_ClassInfoI__delete(
  struct SIDL_ClassInfoI__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_SIDL_ClassInfoI_addRef(
  struct SIDL_ClassInfoI__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_SIDL_ClassInfoI_deleteRef(
  struct SIDL_ClassInfoI__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_SIDL_ClassInfoI_isSame(
  struct SIDL_ClassInfoI__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_SIDL_ClassInfoI_queryInt(
  struct SIDL_ClassInfoI__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_SIDL_ClassInfoI_isType(
  struct SIDL_ClassInfoI__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct SIDL_ClassInfo__object*
remote_SIDL_ClassInfoI_getClassInfo(
  struct SIDL_ClassInfoI__object* self)
{
  return (struct SIDL_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:setName
 */

static void
remote_SIDL_ClassInfoI_setName(
  struct SIDL_ClassInfoI__object* self,
  const char* name)
{
}

/*
 * REMOTE METHOD STUB:setIORVersion
 */

static void
remote_SIDL_ClassInfoI_setIORVersion(
  struct SIDL_ClassInfoI__object* self,
  int32_t major,
  int32_t minor)
{
}

/*
 * REMOTE METHOD STUB:getName
 */

static char*
remote_SIDL_ClassInfoI_getName(
  struct SIDL_ClassInfoI__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getIORVersion
 */

static char*
remote_SIDL_ClassInfoI_getIORVersion(
  struct SIDL_ClassInfoI__object* self)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void SIDL_ClassInfoI__init_remote_epv(void)
{
  struct SIDL_ClassInfoI__epv*    epv = &s_rem__sidl_classinfoi;
  struct SIDL_BaseClass__epv*     e0  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e1  = &s_rem__sidl_baseinterface;
  struct SIDL_ClassInfo__epv*     e2  = &s_rem__sidl_classinfo;

  epv->f__cast         = remote_SIDL_ClassInfoI__cast;
  epv->f__delete       = remote_SIDL_ClassInfoI__delete;
  epv->f__ctor         = NULL;
  epv->f__dtor         = NULL;
  epv->f_addRef        = remote_SIDL_ClassInfoI_addRef;
  epv->f_deleteRef     = remote_SIDL_ClassInfoI_deleteRef;
  epv->f_isSame        = remote_SIDL_ClassInfoI_isSame;
  epv->f_queryInt      = remote_SIDL_ClassInfoI_queryInt;
  epv->f_isType        = remote_SIDL_ClassInfoI_isType;
  epv->f_getClassInfo  = remote_SIDL_ClassInfoI_getClassInfo;
  epv->f_setName       = remote_SIDL_ClassInfoI_setName;
  epv->f_setIORVersion = remote_SIDL_ClassInfoI_setIORVersion;
  epv->f_getName       = remote_SIDL_ClassInfoI_getName;
  epv->f_getIORVersion = remote_SIDL_ClassInfoI_getIORVersion;

  e0->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  e2->f__cast         = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete       = (void (*)(void*)) epv->f__delete;
  e2->f_addRef        = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef     = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame        = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt      = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType        = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo  = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_getName       = (char* (*)(void*)) epv->f_getName;
  e2->f_getIORVersion = (char* (*)(void*)) epv->f_getIORVersion;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct SIDL_ClassInfoI__object*
SIDL_ClassInfoI__remote(const char *url)
{
  struct SIDL_ClassInfoI__object* self =
    (struct SIDL_ClassInfoI__object*) malloc(
      sizeof(struct SIDL_ClassInfoI__object));

  struct SIDL_ClassInfoI__object* s0 = self;
  struct SIDL_BaseClass__object*  s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    SIDL_ClassInfoI__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_sidl_classinfo.d_epv    = &s_rem__sidl_classinfo;
  s0->d_sidl_classinfo.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__sidl_classinfoi;

  return self;
}
