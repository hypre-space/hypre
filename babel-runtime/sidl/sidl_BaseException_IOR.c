/*
 * File:          sidl_BaseException_IOR.c
 * Symbol:        sidl.BaseException-v0.9.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.BaseException
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
#include "sidl_BaseException_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 8;
/*
 * Static variables for managing EPV initialization.
 */

static int s_remote_initialized = 0;

static struct sidl_BaseException__epv s_rem__sidl_baseexception;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_sidl_BaseException__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_sidl_BaseException__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_sidl_BaseException_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_sidl_BaseException_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_sidl_BaseException_isSame(
  void* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_sidl_BaseException_queryInt(
  void* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_sidl_BaseException_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_sidl_BaseException_getClassInfo(
  void* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:getNote
 */

static char*
remote_sidl_BaseException_getNote(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:setNote
 */

static void
remote_sidl_BaseException_setNote(
  void* self,
  const char* message)
{
}

/*
 * REMOTE METHOD STUB:getTrace
 */

static char*
remote_sidl_BaseException_getTrace(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:addLine
 */

static void
remote_sidl_BaseException_addLine(
  void* self,
  const char* traceline)
{
}

/*
 * REMOTE METHOD STUB:add
 */

static void
remote_sidl_BaseException_add(
  void* self,
  const char* filename,
  int32_t lineno,
  const char* methodname)
{
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void sidl_BaseException__init_remote_epv(void)
{
  struct sidl_BaseException__epv* epv = &s_rem__sidl_baseexception;

  epv->f__cast        = remote_sidl_BaseException__cast;
  epv->f__delete      = remote_sidl_BaseException__delete;
  epv->f_addRef       = remote_sidl_BaseException_addRef;
  epv->f_deleteRef    = remote_sidl_BaseException_deleteRef;
  epv->f_isSame       = remote_sidl_BaseException_isSame;
  epv->f_queryInt     = remote_sidl_BaseException_queryInt;
  epv->f_isType       = remote_sidl_BaseException_isType;
  epv->f_getClassInfo = remote_sidl_BaseException_getClassInfo;
  epv->f_getNote      = remote_sidl_BaseException_getNote;
  epv->f_setNote      = remote_sidl_BaseException_setNote;
  epv->f_getTrace     = remote_sidl_BaseException_getTrace;
  epv->f_addLine      = remote_sidl_BaseException_addLine;
  epv->f_add          = remote_sidl_BaseException_add;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct sidl_BaseException__object*
sidl_BaseException__remote(const char *url)
{
  struct sidl_BaseException__object* self =
    (struct sidl_BaseException__object*) malloc(
      sizeof(struct sidl_BaseException__object));

  if (!s_remote_initialized) {
    sidl_BaseException__init_remote_epv();
  }

  self->d_epv    = &s_rem__sidl_baseexception;
  self->d_object = NULL; /* FIXME */

  return self;
}
