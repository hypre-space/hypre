/*
 * File:          SIDL_BaseClass_IOR.c
 * Symbol:        SIDL.BaseClass-v0.7.5
 * Symbol Type:   class
 * Babel Version: 0.7.5
 * Release:       $Name$
 * Revision:      @(#) $Id$
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
 * babel-version = 0.7.5
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "SIDL_BaseClass_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct SIDL_BaseClass__epv s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void SIDL_BaseClass__set_epv(
  struct SIDL_BaseClass__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* SIDL_BaseClass__cast(
  struct SIDL_BaseClass__object* self,
  const char* name)
{
  void* cast = NULL;

  struct SIDL_BaseClass__object* s0 = self;

  if (!strcmp(name, "SIDL.BaseClass")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "SIDL.BaseInterface")) {
    cast = (void*) &s0->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void SIDL_BaseClass__delete(
  struct SIDL_BaseClass__object* self)
{
  SIDL_BaseClass__fini(self);
  memset((void*)self, 0, sizeof(struct SIDL_BaseClass__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void SIDL_BaseClass__init_epv(
  struct SIDL_BaseClass__object* self)
{
  struct SIDL_BaseClass__epv*     epv = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e0  = &s_new__sidl_baseinterface;

  epv->f__cast           = SIDL_BaseClass__cast;
  epv->f__delete         = SIDL_BaseClass__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = NULL;
  epv->f_deleteReference = NULL;
  epv->f_isSame          = NULL;
  epv->f_queryInterface  = NULL;
  epv->f_isInstanceOf    = NULL;

  SIDL_BaseClass__set_epv(epv);

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_method_initialized = 1;
}

/*
 * NEW: allocate object and initialize it.
 */

struct SIDL_BaseClass__object*
SIDL_BaseClass__new(void)
{
  struct SIDL_BaseClass__object* self =
    (struct SIDL_BaseClass__object*) malloc(
      sizeof(struct SIDL_BaseClass__object));
  SIDL_BaseClass__init(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void SIDL_BaseClass__init(
  struct SIDL_BaseClass__object* self)
{
  struct SIDL_BaseClass__object* s0 = self;

  if (!s_method_initialized) {
    SIDL_BaseClass__init_epv(s0);
  }

  s0->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s0->d_epv                      = &s_new__sidl_baseclass;

  s0->d_sidl_baseinterface.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void SIDL_BaseClass__fini(
  struct SIDL_BaseClass__object* self)
{
  struct SIDL_BaseClass__object* s0 = self;

  (*(s0->d_epv->f__dtor))(s0);
}

static const struct SIDL_BaseClass__external
s_externalEntryPoints = {
  SIDL_BaseClass__new,
  SIDL_BaseClass__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct SIDL_BaseClass__external*
SIDL_BaseClass__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_SIDL_BaseClass__cast(
  struct SIDL_BaseClass__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_SIDL_BaseClass__delete(
  struct SIDL_BaseClass__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_SIDL_BaseClass_addReference(
  struct SIDL_BaseClass__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_SIDL_BaseClass_deleteReference(
  struct SIDL_BaseClass__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_SIDL_BaseClass_isSame(
  struct SIDL_BaseClass__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_SIDL_BaseClass_queryInterface(
  struct SIDL_BaseClass__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_SIDL_BaseClass_isInstanceOf(
  struct SIDL_BaseClass__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void SIDL_BaseClass__init_remote_epv(void)
{
  struct SIDL_BaseClass__epv*     epv = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e0  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_SIDL_BaseClass__cast;
  epv->f__delete         = remote_SIDL_BaseClass__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = remote_SIDL_BaseClass_addReference;
  epv->f_deleteReference = remote_SIDL_BaseClass_deleteReference;
  epv->f_isSame          = remote_SIDL_BaseClass_isSame;
  epv->f_queryInterface  = remote_SIDL_BaseClass_queryInterface;
  epv->f_isInstanceOf    = remote_SIDL_BaseClass_isInstanceOf;

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct SIDL_BaseClass__object*
SIDL_BaseClass__remote(const char *url)
{
  struct SIDL_BaseClass__object* self =
    (struct SIDL_BaseClass__object*) malloc(
      sizeof(struct SIDL_BaseClass__object));

  struct SIDL_BaseClass__object* s0 = self;

  if (!s_remote_initialized) {
    SIDL_BaseClass__init_remote_epv();
  }

  s0->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__sidl_baseclass;

  return self;
}
