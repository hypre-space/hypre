/*
 * File:          sidl_BaseClass_Impl.c
 * Symbol:        sidl.BaseClass-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V1-14-0b $
 * Revision:      @(#) $Id: sidl_BaseClass_Impl.c,v 1.7 2006/08/29 22:29:49 painter Exp $
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
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.BaseClass" (version 0.9.15)
 * 
 * Every class implicitly inherits from <code>BaseClass</code>.  This
 * class implements the methods in <code>BaseInterface</code>.
 */

#include "sidl_BaseClass_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidl.BaseClass._includes) */
#include <stdlib.h>
#include "sidl_BaseInterface.h"
#include <stdio.h>
#include "sidl_rmi_InstanceRegistry.h"
#include "sidl_Exception.h"
#include "sidlOps.h"

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
sidl_report_objects(void *ignored)
{
  sidl_BaseInterface ex = NULL;
  struct sidl_BaseClass_list *ptr = s_object_list;
  if (ptr) {
    do {
      char *type = NULL;
      struct sidl_BaseClass__data *data = sidl_BaseClass__get_data(ptr->d_obj);
      if (data->d_classinfo) {
        type = sidl_ClassInfo_getName(data->d_classinfo, &ex);
	if(ex) {  //If there's an exception, handle it.
	  sidl_BaseException s_b_e = NULL;
	  sidl_BaseInterface throwaway = NULL;
	  char* str = NULL;
	  s_b_e = sidl_BaseException__cast(ex, &throwaway);
	  if(throwaway != NULL) {
	    fprintf(stderr, "babel: Exception occured and was uncatchable\n");
	    if (type) free((void *)type);
	    ptr = ptr->d_next;
	    continue;
	  }
	  str = sidl_BaseException_getNote(s_b_e, &throwaway);
	  if(throwaway != NULL) {
	    fprintf(stderr, "babel: Exception occured and was uncatchable unprintable\n");
	    if (type) free((void *)type);
	    ptr = ptr->d_next;
	    continue;
	  }
	  printf("babel: sidl_report_objects: %s \n",str);
	  if (type) free((void *)type);
	  ptr = ptr->d_next;
	  continue;
	}
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
  return;
}

static void
sidl_initialize_list(void)
{
  static int s_not_initialized = 1;
  if (s_not_initialized) {
    s_not_initialized = 0;
    sidl_atexit(sidl_report_objects, NULL);
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

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_BaseClass__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_BaseClass__ctor(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
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
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_BaseClass__ctor2(
  /* in */ sidl_BaseClass self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass._ctor2) */
  /* Insert-Code-Here {sidl.BaseClass._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_BaseClass__dtor(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass._dtor) */
  struct sidl_BaseClass__data *data = sidl_BaseClass__get_data(self);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl_remove_object(self);
#endif
  sidl_BaseClass__set_data(self, NULL);
  if (data) {
    sidl_BaseInterface bi = (sidl_BaseInterface)data->d_classinfo;
    data->d_classinfo = NULL;
    if (bi) {
      sidl_BaseInterface_deleteRef(bi, _ex);
    }
#ifdef HAVE_PTHREAD
    (void)pthread_mutex_destroy(&(data->d_mutex));
#endif
    free((void*) data);
  }
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass._dtor) */
  }
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_BaseClass_addRef(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.addRef) */
  struct sidl_BaseClass__data* data = sidl_BaseClass__get_data(self);
  
   if (data) {
#ifdef SIDL_DEBUG_REFCOUNT
     char *type = NULL;
#endif
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_lock(&(data->d_mutex));
#endif /* HAVE_PTHREAD */
     if (data->d_refcount > 0) { 
       /* only addRef is reference count is positive */
       ++(data->d_refcount);
     }
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_unlock(&(data->d_mutex));
#endif /* HAVE_PTHREAD */
#ifdef SIDL_DEBUG_REFCOUNT
     if (data->d_classinfo) {
       type = sidl_ClassInfo_getName(data->d_classinfo, _ex); SIDL_CHECK(*_ex);
     }
     fprintf(stderr, "babel: addRef %p new count %d (type %s)\n",
             self, data->d_refcount, (type ? type : ""));
     if (type) free((void *)type);
#endif
   }
 EXIT:
   return;
   /* DO-NOT-DELETE splicer.end(sidl.BaseClass.addRef) */
  }
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_BaseClass_deleteRef(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
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
     self_destruct = ((--(data->d_refcount)) == 0);
#ifdef SIDL_DEBUG_REFCOUNT
     if (data->d_classinfo) {
       type = sidl_ClassInfo_getName(data->d_classinfo, _ex); SIDL_CHECK(*_ex);
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
     char* objID = sidl_rmi_InstanceRegistry_removeInstanceByClass(self, _ex); SIDL_CHECK(*_ex);
     sidl_BaseClass__delete(self,_ex); SIDL_CHECK(*_ex);
     free((void*)objID);
   }
 EXIT:
   return;
   /* DO-NOT-DELETE splicer.end(sidl.BaseClass.deleteRef) */
  }
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass_isSame"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidl_BaseClass_isSame(
  /* in */ sidl_BaseClass self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.isSame) */
  const sidl_BaseClass bg = sidl_BaseClass__cast(iobj,_ex);
  const sidl_bool result = (self == bg);
  SIDL_CHECK(*_ex);
  if (bg) sidl_BaseClass_deleteRef(bg,_ex); SIDL_CHECK(*_ex);
 EXIT:
  return result;
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass.isSame) */
  }
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass_isType"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidl_BaseClass_isType(
  /* in */ sidl_BaseClass self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.isType) */
  const sidl_BaseInterface bi = (sidl_BaseInterface)
    sidl_BaseClass__cast2(self, name, _ex);
  const sidl_bool result = bi ? TRUE : FALSE;
  if (bi) sidl_BaseInterface_deleteRef(bi, _ex);
  return result;
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass.isType) */
  }
}

/*
 * Return the meta-data about the class implementing this interface.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_BaseClass_getClassInfo"

#ifdef __cplusplus
extern "C"
#endif
sidl_ClassInfo
impl_sidl_BaseClass_getClassInfo(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass.getClassInfo) */
   struct sidl_BaseClass__data* data = sidl_BaseClass__get_data(self);
   if (data) {
     sidl_BaseInterface bi = (sidl_BaseInterface)data->d_classinfo;
     if (bi) {
       sidl_BaseInterface_addRef(bi,_ex);
       return data->d_classinfo;
     }
   }
   return NULL;
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass.getClassInfo) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidl_BaseClass_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_sidl_BaseClass_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_BaseClass_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_BaseClass_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidl_BaseClass_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_sidl_BaseClass_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_BaseClass_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_BaseClass_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
