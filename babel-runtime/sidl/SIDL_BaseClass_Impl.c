/*
 * File:          SIDL_BaseClass_Impl.c
 * Symbol:        SIDL.BaseClass-v0.7.5
 * Symbol Type:   class
 * Babel Version: 0.7.5
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for SIDL.BaseClass
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
 * babel-version = 0.7.5
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "SIDL.BaseClass" (version 0.7.5)
 * 
 * Every class implicitly inherits from <code>BaseClass</code>.  This
 * class implements the methods in <code>BaseInterface</code>.
 */

#include "SIDL_BaseClass_Impl.h"

/* DO-NOT-DELETE splicer.begin(SIDL.BaseClass._includes) */
#include <stdlib.h>
#include "SIDL_BaseInterface.h"

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef NULL
#define NULL 0
#endif
/* DO-NOT-DELETE splicer.end(SIDL.BaseClass._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseClass__ctor"

void
impl_SIDL_BaseClass__ctor(
  SIDL_BaseClass self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseClass._ctor) */
  struct SIDL_BaseClass__data *data = (struct SIDL_BaseClass__data *)
    malloc(sizeof (struct SIDL_BaseClass__data));
  data->d_refcount = 1;
  SIDL_BaseClass__set_data(self, data);
  /* DO-NOT-DELETE splicer.end(SIDL.BaseClass._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseClass__dtor"

void
impl_SIDL_BaseClass__dtor(
  SIDL_BaseClass self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseClass._dtor) */
  struct SIDL_BaseClass__data *data = SIDL_BaseClass__get_data(self);
  if (data) {
    free((void*) data);
  }
  SIDL_BaseClass__set_data(self, NULL);
  /* DO-NOT-DELETE splicer.end(SIDL.BaseClass._dtor) */
}

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>SIDL</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseClass_addReference"

void
impl_SIDL_BaseClass_addReference(
  SIDL_BaseClass self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseClass.addReference) */
   struct SIDL_BaseClass__data* data = SIDL_BaseClass__get_data(self);
   if (data) {
     ++(data->d_refcount);
   }
   /* DO-NOT-DELETE splicer.end(SIDL.BaseClass.addReference) */
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseClass_deleteReference"

void
impl_SIDL_BaseClass_deleteReference(
  SIDL_BaseClass self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseClass.deleteReference) */
   struct SIDL_BaseClass__data* data = SIDL_BaseClass__get_data(self);
   int self_destruct = TRUE;
   if (data) {
     self_destruct = ((--(data->d_refcount)) <= 0);
   }
   if (self_destruct) {
     SIDL_BaseClass__delete(self);
   }
   /* DO-NOT-DELETE splicer.end(SIDL.BaseClass.deleteReference) */
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseClass_isSame"

SIDL_bool
impl_SIDL_BaseClass_isSame(
  SIDL_BaseClass self, SIDL_BaseInterface iobj)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseClass.isSame) */
  return self == SIDL_BaseClass__cast(iobj);
  /* DO-NOT-DELETE splicer.end(SIDL.BaseClass.isSame) */
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteReference</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseClass_queryInterface"

SIDL_BaseInterface
impl_SIDL_BaseClass_queryInterface(
  SIDL_BaseClass self, const char* name)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseClass.queryInterface) */
  SIDL_BaseInterface result = 
    (SIDL_BaseInterface)SIDL_BaseInterface__cast2(self, name);
  if (result) {
    SIDL_BaseInterface_addReference(result);
  }
  return result;
  /* DO-NOT-DELETE splicer.end(SIDL.BaseClass.queryInterface) */
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseClass_isInstanceOf"

SIDL_bool
impl_SIDL_BaseClass_isInstanceOf(
  SIDL_BaseClass self, const char* name)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseClass.isInstanceOf) */
  return SIDL_BaseClass__cast2(self, name) ? TRUE : FALSE;
  /* DO-NOT-DELETE splicer.end(SIDL.BaseClass.isInstanceOf) */
}
