/*
 * File:          SIDL_BaseInterface_IOR.c
 * Symbol:        SIDL.BaseInterface-v0.7.5
 * Symbol Type:   interface
 * Babel Version: 0.7.5
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for SIDL.BaseInterface
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
#include "SIDL_BaseInterface_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_remote_initialized = 0;

static struct SIDL_BaseInterface__epv s_rem__sidl_baseinterface;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_SIDL_BaseInterface__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_SIDL_BaseInterface__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_SIDL_BaseInterface_addReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_SIDL_BaseInterface_deleteReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_SIDL_BaseInterface_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_SIDL_BaseInterface_queryInterface(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_SIDL_BaseInterface_isInstanceOf(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void SIDL_BaseInterface__init_remote_epv(void)
{
  struct SIDL_BaseInterface__epv* epv = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_SIDL_BaseInterface__cast;
  epv->f__delete         = remote_SIDL_BaseInterface__delete;
  epv->f_addReference    = remote_SIDL_BaseInterface_addReference;
  epv->f_deleteReference = remote_SIDL_BaseInterface_deleteReference;
  epv->f_isSame          = remote_SIDL_BaseInterface_isSame;
  epv->f_queryInterface  = remote_SIDL_BaseInterface_queryInterface;
  epv->f_isInstanceOf    = remote_SIDL_BaseInterface_isInstanceOf;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct SIDL_BaseInterface__object*
SIDL_BaseInterface__remote(const char *url)
{
  struct SIDL_BaseInterface__object* self =
    (struct SIDL_BaseInterface__object*) malloc(
      sizeof(struct SIDL_BaseInterface__object));

  if (!s_remote_initialized) {
    SIDL_BaseInterface__init_remote_epv();
  }

  self->d_epv    = &s_rem__sidl_baseinterface;
  self->d_object = NULL; /* FIXME */

  return self;
}
