/*
 * File:          sidl_BaseClass_Impl.c
 * Symbol:        sidl.BaseClass-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.BaseClass
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
 * babel-version = 0.9.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.BaseClass" (version 0.9.0)
 * 
 * Every class implicitly inherits from <code>BaseClass</code>.  This
 * class implements the methods in <code>BaseInterface</code>.
 */

#include "sidl_BaseClass_Impl.h"

#line 52 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidl.BaseClass._includes) */
#include <stdlib.h>
#include "sidl_BaseInterface.h"
#include <stdio.h>

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef NULL
#define NULL 0
#endif

#ifdef SIDL_DEBUG_REFCOUNT
#include <stdio.h>
#include <stdlib.h>

struct sidl_BaseClass_list {
  struct sidl_BaseClass_list *d_next;
  sidl_BaseClass              d_obj;
};

static struct sidl_BaseClass_list *s_object_list = NULL;

static void
sidl_report_objects()
{
  struct sidl_BaseClass_list *ptr = s_object_list;
  if (ptr) {
    do {
      char *type = NULL;
      struct sidl_BaseClass__data *data = sidl_BaseClass__get_data(ptr->d_obj);
      if (data->d_classinfo) {
        type = sidl_ClassInfo_getName(data->d_classinfo);
      }
      fprintf(stderr, "babel: leaked object %p reference count %d (type %s)\n", 
              ptr->d_obj, data->d_refcount, (type ? type : NULL));
      if (type) free((void *)type);
      ptr = ptr->d_next;
    } while (ptr);
  }
  else {
    fprintf(stderr, "babel: no objects leaked\n");
  }
}

static void
sidl_initialize_list()
{
  static int s_not_initialized = 1;
  if (s_not_initialized) {
    s_not_initialized = 0;
    atexit(sidl_report_objects);
  }
}

static void
sidl_add_object(sidl_BaseClass cls)
{
  sidl_initialize_list();
  if (cls){
    struct sidl_BaseClass_list *ptr = 
      malloc(sizeof(struct sidl_BaseClass_list));
    ptr->d_next = s_object_list;
    ptr->d_obj = cls;
    s_object_list = ptr;
  }
}

static void
sidl_remove_object(sidl_BaseClass cls)
{
  sidl_initialize_list();
  if (cls) {
    struct sidl_BaseClass_list *prev, *ptr;
    if (s_object_list && (s_object_list->d_obj == cls)) {
      ptr = s_object_list->d_next;
      free((void *)s_object_list);
      s_object_list = ptr;
    }
    else {
      prev = s_object_list;
      ptr = (prev ? prev->d_next : NULL);
      while (ptr) {
        if (ptr->d_obj == cls) {
          struct sidl_BaseClass_list *next = ptr->d_next;
          free((void *)ptr);
          prev->d_next = next;
          return;
        }
        prev = ptr;
        ptr = ptr->d_next;
      }
      fprintf(stderr, "babel: data type invariant failure %p\n", cls);
    }
  }
}
#endif /* SIDL_DEBUG_REFCOUNT */
/* DO-NOT-DELETE splicer.end(sidl.BaseClass._includes) */
#line 154 "sidl_BaseClass_Impl.c"

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass__ctor"

void
impl_sidl_BaseClass__ctor(
  /*in*/ sidl_BaseClass self)
{
#line 165 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass._ctor) */
  struct sidl_BaseClass__data *data = (struct sidl_BaseClass__data *)
    malloc(sizeof (struct sidl_BaseClass__data));
  data->d_refcount = 1;
  data->d_classinfo = NULL;
  data->d_IOR_major_version = -1;
  data->d_IOR_minor_version = -1;
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_init(&(data->d_mutex), NULL);
#endif /* HAVE_PTHREAD */
  sidl_BaseClass__set_data(self, data);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl_add_object(self);
#endif
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass._ctor) */
#line 183 "sidl_BaseClass_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass__dtor"

void
impl_sidl_BaseClass__dtor(
  /*in*/ sidl_BaseClass self)
{
#line 193 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass._dtor) */
  struct sidl_BaseClass__data *data = sidl_BaseClass__get_data(self);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl_remove_object(self);
#endif
  if (data) {
    sidl_BaseInterface bi = (sidl_BaseInterface)data->d_classinfo;
    if (bi) {
      sidl_BaseInterface_deleteRef(bi);
    }
#ifdef HAVE_PTHREAD
    (void)pthread_mutex_destroy(&(data->d_mutex));
#endif
    free((void*) data);
  }
  sidl_BaseClass__set_data(self, NULL);
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass._dtor) */
#line 215 "sidl_BaseClass_Impl.c"
}

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
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
#define __FUNC__ "impl_sidl_BaseClass_addRef"

void
impl_sidl_BaseClass_addRef(
  /*in*/ sidl_BaseClass self)
{
#line 234 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.addRef) */
  struct sidl_BaseClass__data* data = sidl_BaseClass__get_data(self);
  
   if (data) {
#ifdef SIDL_DEBUG_REFCOUNT
     char *type = NULL;
#endif
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_lock(&(data->d_mutex));
#endif /* HAVE_PTHREAD */
     ++(data->d_refcount);
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_unlock(&(data->d_mutex));
#endif /* HAVE_PTHREAD */
#ifdef SIDL_DEBUG_REFCOUNT
     if (data->d_classinfo) {
       type = sidl_ClassInfo_getName(data->d_classinfo);
     }
     fprintf(stderr, "babel: addRef %p new count %d (type %s)\n",
             self, data->d_refcount, (type ? type : ""));
     if (type) free((void *)type);
#endif
   }
   /* DO-NOT-DELETE splicer.end(sidl.BaseClass.addRef) */
#line 265 "sidl_BaseClass_Impl.c"
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass_deleteRef"

void
impl_sidl_BaseClass_deleteRef(
  /*in*/ sidl_BaseClass self)
{
#line 275 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.deleteRef) */
   struct sidl_BaseClass__data* data = sidl_BaseClass__get_data(self);
   int self_destruct = TRUE;
   if (data) {
#ifdef SIDL_DEBUG_REFCOUNT
     char *type = NULL;
     int  refcount;
#endif
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_lock(&(data->d_mutex));
#endif /* HAVE_PTHREAD */
     self_destruct = ((--(data->d_refcount)) <= 0);
#ifdef SIDL_DEBUG_REFCOUNT
     if (data->d_classinfo) {
       type = sidl_ClassInfo_getName(data->d_classinfo);
     }
     refcount = data->d_refcount;
#endif
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_unlock(&(data->d_mutex));
#endif /* HAVE_PTHREAD */
#ifdef SIDL_DEBUG_REFCOUNT
     fprintf(stderr, "babel: deleteRef %p new count %d (type %s)\n",
             self, refcount, (type ? type : ""));
     if (type) free((void *)type);
#endif
   }
   if (self_destruct) {
     sidl_BaseClass__delete(self);
   }
   /* DO-NOT-DELETE splicer.end(sidl.BaseClass.deleteRef) */
#line 315 "sidl_BaseClass_Impl.c"
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass_isSame"

sidl_bool
impl_sidl_BaseClass_isSame(
  /*in*/ sidl_BaseClass self, /*in*/ sidl_BaseInterface iobj)
{
#line 320 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.isSame) */
  return self == sidl_BaseClass__cast(iobj);
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass.isSame) */
#line 334 "sidl_BaseClass_Impl.c"
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass_queryInt"

sidl_BaseInterface
impl_sidl_BaseClass_queryInt(
  /*in*/ sidl_BaseClass self, /*in*/ const char* name)
{
#line 342 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.queryInt) */
  sidl_BaseInterface result = 
    (sidl_BaseInterface)sidl_BaseInterface__cast2(self, name);
  if (result) {
    sidl_BaseInterface_addRef(result);
  }
  return result;
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass.queryInt) */
#line 363 "sidl_BaseClass_Impl.c"
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass_isType"

sidl_bool
impl_sidl_BaseClass_isType(
  /*in*/ sidl_BaseClass self, /*in*/ const char* name)
{
#line 366 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.isType) */
  return sidl_BaseClass__cast2(self, name) ? TRUE : FALSE;
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass.isType) */
#line 384 "sidl_BaseClass_Impl.c"
}

/*
 * Return the meta-data about the class implementing this interface.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass_getClassInfo"

sidl_ClassInfo
impl_sidl_BaseClass_getClassInfo(
  /*in*/ sidl_BaseClass self)
{
#line 382 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.getClassInfo) */
   struct sidl_BaseClass__data* data = sidl_BaseClass__get_data(self);
   if (data) {
     sidl_BaseInterface bi = (sidl_BaseInterface)data->d_classinfo;
     if (bi) {
       sidl_BaseInterface_addRef(bi);
       return data->d_classinfo;
     }
   }
   return NULL;
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass.getClassInfo) */
#line 410 "sidl_BaseClass_Impl.c"
}
