/*
 * File:          SIDL_Loader_IOR.c
 * Symbol:        SIDL.Loader-v0.7.5
 * Symbol Type:   class
 * Babel Version: 0.7.5
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
 * babel-version = 0.7.5
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "SIDL_Loader_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;
static int s_static_initialized = 0;

static struct SIDL_Loader__epv  s_new__sidl_loader;
static struct SIDL_Loader__epv  s_rem__sidl_loader;
static struct SIDL_Loader__sepv s_stc__sidl_loader;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void SIDL_Loader__set_epv(
  struct SIDL_Loader__epv* epv);
extern void SIDL_Loader__set_sepv(
  struct SIDL_Loader__sepv* sepv);

/*
 * CAST: dynamic type casting support.
 */

static void* SIDL_Loader__cast(
  struct SIDL_Loader__object* self,
  const char* name)
{
  void* cast = NULL;

  struct SIDL_Loader__object*    s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "SIDL.Loader")) {
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

static void SIDL_Loader__delete(
  struct SIDL_Loader__object* self)
{
  SIDL_Loader__fini(self);
  memset((void*)self, 0, sizeof(struct SIDL_Loader__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void SIDL_Loader__init_epv(
  struct SIDL_Loader__object* self)
{
  struct SIDL_Loader__object*    s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  struct SIDL_Loader__epv*        epv = &s_new__sidl_loader;
  struct SIDL_BaseClass__epv*     e0  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e1  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = SIDL_Loader__cast;
  epv->f__delete         = SIDL_Loader__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = (void (*)(struct SIDL_Loader__object*)) 
    s1->d_epv->f_addReference;
  epv->f_deleteReference = (void (*)(struct SIDL_Loader__object*)) 
    s1->d_epv->f_deleteReference;
  epv->f_isSame          = (SIDL_bool (*)(struct SIDL_Loader__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_Loader__object*,const char*)) s1->d_epv->f_queryInterface;
  epv->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_Loader__object*,
    const char*)) s1->d_epv->f_isInstanceOf;

  SIDL_Loader__set_epv(epv);

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
 * SEPV: create the static entry point vector (SEPV).
 */

static void SIDL_Loader__init_sepv(void)
{
  struct SIDL_Loader__sepv* s = &s_stc__sidl_loader;

  s->f_setSearchPath   = NULL;
  s->f_getSearchPath   = NULL;
  s->f_addSearchPath   = NULL;
  s->f_loadLibrary     = NULL;
  s->f_addDLL          = NULL;
  s->f_unloadLibraries = NULL;
  s->f_lookupSymbol    = NULL;
  s->f_createClass     = NULL;

  SIDL_Loader__set_sepv(s);

  s_static_initialized = 1;
}

/*
 * STATIC: return static EPV structure for static methods.
 */

struct SIDL_Loader__sepv*
SIDL_Loader__statics(void)
{
  if (!s_static_initialized) {
    SIDL_Loader__init_sepv();
  }
  return &s_stc__sidl_loader;
}

/*
 * NEW: allocate object and initialize it.
 */

struct SIDL_Loader__object*
SIDL_Loader__new(void)
{
  struct SIDL_Loader__object* self =
    (struct SIDL_Loader__object*) malloc(
      sizeof(struct SIDL_Loader__object));
  SIDL_Loader__init(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void SIDL_Loader__init(
  struct SIDL_Loader__object* self)
{
  struct SIDL_Loader__object*    s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    SIDL_Loader__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_epv    = &s_new__sidl_loader;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void SIDL_Loader__fini(
  struct SIDL_Loader__object* self)
{
  struct SIDL_Loader__object*    s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

static const struct SIDL_Loader__external
s_externalEntryPoints = {
  SIDL_Loader__new,
  SIDL_Loader__remote,
  SIDL_Loader__statics,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct SIDL_Loader__external*
SIDL_Loader__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_SIDL_Loader__cast(
  struct SIDL_Loader__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_SIDL_Loader__delete(
  struct SIDL_Loader__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_SIDL_Loader_addReference(
  struct SIDL_Loader__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_SIDL_Loader_deleteReference(
  struct SIDL_Loader__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_SIDL_Loader_isSame(
  struct SIDL_Loader__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_SIDL_Loader_queryInterface(
  struct SIDL_Loader__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_SIDL_Loader_isInstanceOf(
  struct SIDL_Loader__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void SIDL_Loader__init_remote_epv(void)
{
  struct SIDL_Loader__epv*        epv = &s_rem__sidl_loader;
  struct SIDL_BaseClass__epv*     e0  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e1  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_SIDL_Loader__cast;
  epv->f__delete         = remote_SIDL_Loader__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = remote_SIDL_Loader_addReference;
  epv->f_deleteReference = remote_SIDL_Loader_deleteReference;
  epv->f_isSame          = remote_SIDL_Loader_isSame;
  epv->f_queryInterface  = remote_SIDL_Loader_queryInterface;
  epv->f_isInstanceOf    = remote_SIDL_Loader_isInstanceOf;

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

struct SIDL_Loader__object*
SIDL_Loader__remote(const char *url)
{
  struct SIDL_Loader__object* self =
    (struct SIDL_Loader__object*) malloc(
      sizeof(struct SIDL_Loader__object));

  struct SIDL_Loader__object*    s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    SIDL_Loader__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__sidl_loader;

  return self;
}
