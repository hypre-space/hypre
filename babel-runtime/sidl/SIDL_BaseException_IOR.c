/*
 * File:          SIDL_BaseException_IOR.c
 * Symbol:        SIDL.BaseException-v0.7.5
 * Symbol Type:   class
 * Babel Version: 0.7.5
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for SIDL.BaseException
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
 * babel-version = 0.7.5
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "SIDL_BaseException_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct SIDL_BaseException__epv s_new__sidl_baseexception;
static struct SIDL_BaseException__epv s_rem__sidl_baseexception;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void SIDL_BaseException__set_epv(
  struct SIDL_BaseException__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* SIDL_BaseException__cast(
  struct SIDL_BaseException__object* self,
  const char* name)
{
  void* cast = NULL;

  struct SIDL_BaseException__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "SIDL.BaseException")) {
    cast = (void*) s0;
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

static void SIDL_BaseException__delete(
  struct SIDL_BaseException__object* self)
{
  SIDL_BaseException__fini(self);
  memset((void*)self, 0, sizeof(struct SIDL_BaseException__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void SIDL_BaseException__init_epv(
  struct SIDL_BaseException__object* self)
{
  struct SIDL_BaseException__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  struct SIDL_BaseException__epv* epv = &s_new__sidl_baseexception;
  struct SIDL_BaseClass__epv*     e0  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e1  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = SIDL_BaseException__cast;
  epv->f__delete         = SIDL_BaseException__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = (void (*)(struct SIDL_BaseException__object*)) 
    s1->d_epv->f_addReference;
  epv->f_deleteReference = (void (*)(struct SIDL_BaseException__object*)) 
    s1->d_epv->f_deleteReference;
  epv->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseException__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseException__object*,const char*)) s1->d_epv->f_queryInterface;
  epv->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseException__object*,
    const char*)) s1->d_epv->f_isInstanceOf;
  epv->f_getMessage      = NULL;
  epv->f_setMessage      = NULL;
  epv->f_getTrace        = NULL;
  epv->f_addToStackTrace = NULL;
  epv->f_addToTrace      = NULL;

  SIDL_BaseException__set_epv(epv);

  e0->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e0->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e1->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_method_initialized = 1;
}

/*
 * NEW: allocate object and initialize it.
 */

struct SIDL_BaseException__object*
SIDL_BaseException__new(void)
{
  struct SIDL_BaseException__object* self =
    (struct SIDL_BaseException__object*) malloc(
      sizeof(struct SIDL_BaseException__object));
  SIDL_BaseException__init(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void SIDL_BaseException__init(
  struct SIDL_BaseException__object* self)
{
  struct SIDL_BaseException__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    SIDL_BaseException__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_epv    = &s_new__sidl_baseexception;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void SIDL_BaseException__fini(
  struct SIDL_BaseException__object* self)
{
  struct SIDL_BaseException__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

static const struct SIDL_BaseException__external
s_externalEntryPoints = {
  SIDL_BaseException__new,
  SIDL_BaseException__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct SIDL_BaseException__external*
SIDL_BaseException__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_SIDL_BaseException__cast(
  struct SIDL_BaseException__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_SIDL_BaseException__delete(
  struct SIDL_BaseException__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_SIDL_BaseException_addReference(
  struct SIDL_BaseException__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_SIDL_BaseException_deleteReference(
  struct SIDL_BaseException__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_SIDL_BaseException_isSame(
  struct SIDL_BaseException__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_SIDL_BaseException_queryInterface(
  struct SIDL_BaseException__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_SIDL_BaseException_isInstanceOf(
  struct SIDL_BaseException__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getMessage
 */

static char*
remote_SIDL_BaseException_getMessage(
  struct SIDL_BaseException__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:setMessage
 */

static void
remote_SIDL_BaseException_setMessage(
  struct SIDL_BaseException__object* self,
  const char* message)
{
}

/*
 * REMOTE METHOD STUB:getTrace
 */

static char*
remote_SIDL_BaseException_getTrace(
  struct SIDL_BaseException__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:addToStackTrace
 */

static void
remote_SIDL_BaseException_addToStackTrace(
  struct SIDL_BaseException__object* self,
  const char* traceline)
{
}

/*
 * REMOTE METHOD STUB:addToTrace
 */

static void
remote_SIDL_BaseException_addToTrace(
  struct SIDL_BaseException__object* self,
  const char* filename,
  int32_t lineno,
  const char* methodname)
{
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void SIDL_BaseException__init_remote_epv(void)
{
  struct SIDL_BaseException__epv* epv = &s_rem__sidl_baseexception;
  struct SIDL_BaseClass__epv*     e0  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e1  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_SIDL_BaseException__cast;
  epv->f__delete         = remote_SIDL_BaseException__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = remote_SIDL_BaseException_addReference;
  epv->f_deleteReference = remote_SIDL_BaseException_deleteReference;
  epv->f_isSame          = remote_SIDL_BaseException_isSame;
  epv->f_queryInterface  = remote_SIDL_BaseException_queryInterface;
  epv->f_isInstanceOf    = remote_SIDL_BaseException_isInstanceOf;
  epv->f_getMessage      = remote_SIDL_BaseException_getMessage;
  epv->f_setMessage      = remote_SIDL_BaseException_setMessage;
  epv->f_getTrace        = remote_SIDL_BaseException_getTrace;
  epv->f_addToStackTrace = remote_SIDL_BaseException_addToStackTrace;
  epv->f_addToTrace      = remote_SIDL_BaseException_addToTrace;

  e0->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e0->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e1->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct SIDL_BaseException__object*
SIDL_BaseException__remote(const char *url)
{
  struct SIDL_BaseException__object* self =
    (struct SIDL_BaseException__object*) malloc(
      sizeof(struct SIDL_BaseException__object));

  struct SIDL_BaseException__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    SIDL_BaseException__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__sidl_baseexception;

  return self;
}
