/*
 * File:          SIDL_ClassInfoI_Impl.c
 * Symbol:        SIDL.ClassInfoI-v0.8.1
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for SIDL.ClassInfoI
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
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "SIDL.ClassInfoI" (version 0.8.1)
 * 
 * An implementation of the <code>ClassInfo</code> interface. This provides
 * methods to set all the attributes that are read-only in the
 * <code>ClassInfo</code> interface.
 */

#include "SIDL_ClassInfoI_Impl.h"

/* DO-NOT-DELETE splicer.begin(SIDL.ClassInfoI._includes) */
#include <stdlib.h>
#include "SIDL_String.h"
/* DO-NOT-DELETE splicer.end(SIDL.ClassInfoI._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_ClassInfoI__ctor"

void
impl_SIDL_ClassInfoI__ctor(
  SIDL_ClassInfoI self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.ClassInfoI._ctor) */
  struct SIDL_ClassInfoI__data *data = (struct SIDL_ClassInfoI__data*)
    malloc(sizeof(struct SIDL_ClassInfoI__data));
  if (data) {
    data->d_IOR_major = data->d_IOR_minor = -1;
    data->d_classname = NULL;
  }
  SIDL_ClassInfoI__set_data(self, data);
  /* DO-NOT-DELETE splicer.end(SIDL.ClassInfoI._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_ClassInfoI__dtor"

void
impl_SIDL_ClassInfoI__dtor(
  SIDL_ClassInfoI self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.ClassInfoI._dtor) */
  struct SIDL_ClassInfoI__data *data = SIDL_ClassInfoI__get_data(self);
  if (data) {
    SIDL_String_free(data->d_classname);
    free((void *)data);
  }
  /* DO-NOT-DELETE splicer.end(SIDL.ClassInfoI._dtor) */
}

/*
 * Set the name of the class.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_ClassInfoI_setName"

void
impl_SIDL_ClassInfoI_setName(
  SIDL_ClassInfoI self, const char* name)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.ClassInfoI.setName) */
  struct SIDL_ClassInfoI__data *data = SIDL_ClassInfoI__get_data(self);
  if (data) {
    SIDL_String_free(data->d_classname);
    data->d_classname = SIDL_String_strdup(name);
  }
  /* DO-NOT-DELETE splicer.end(SIDL.ClassInfoI.setName) */
}

/*
 * Set the IOR major and minor version numbers.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_ClassInfoI_setIORVersion"

void
impl_SIDL_ClassInfoI_setIORVersion(
  SIDL_ClassInfoI self, int32_t major, int32_t minor)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.ClassInfoI.setIORVersion) */
  struct SIDL_ClassInfoI__data *data = SIDL_ClassInfoI__get_data(self);
  if (data) {
    data->d_IOR_major = major;
    data->d_IOR_minor = minor;
  }

  /* DO-NOT-DELETE splicer.end(SIDL.ClassInfoI.setIORVersion) */
}

/*
 * Return the name of the class.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_ClassInfoI_getName"

char*
impl_SIDL_ClassInfoI_getName(
  SIDL_ClassInfoI self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.ClassInfoI.getName) */
  struct SIDL_ClassInfoI__data *data = SIDL_ClassInfoI__get_data(self);
  return SIDL_String_strdup(data ? data->d_classname : NULL);
  /* DO-NOT-DELETE splicer.end(SIDL.ClassInfoI.getName) */
}

/*
 * Get the version of the intermediate object representation.
 * This will be in the form of major_version.minor_version.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_ClassInfoI_getIORVersion"

char*
impl_SIDL_ClassInfoI_getIORVersion(
  SIDL_ClassInfoI self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.ClassInfoI.getIORVersion) */
  int32_t major, minor;
  char buf[34];
  struct SIDL_ClassInfoI__data *data = SIDL_ClassInfoI__get_data(self);
  major = (data ? data->d_IOR_major : -1);
  minor = (data ? data->d_IOR_minor : -1);
  sprintf(buf, "%d.%d", major, minor);
  return SIDL_String_strdup(buf);
  /* DO-NOT-DELETE splicer.end(SIDL.ClassInfoI.getIORVersion) */
}
