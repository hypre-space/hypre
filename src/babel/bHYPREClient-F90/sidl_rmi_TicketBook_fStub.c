/*
 * File:          sidl_rmi_TicketBook_fStub.c
 * Symbol:        sidl.rmi.TicketBook-v0.9.15
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_rmi_TicketBook_fStub.c,v 1.1 2007/02/06 01:23:14 painter Exp $
 * Description:   Client-side glue code for sidl.rmi.TicketBook
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
 */

/*
 * Symbol "sidl.rmi.TicketBook" (version 0.9.15)
 * 
 * This is a collection of Tickets that itself can be viewed
 * as a ticket.  
 */

#ifndef included_sidl_rmi_TicketBook_fStub_h
#include "sidl_rmi_TicketBook_fStub.h"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "sidlfortran.h"
#ifndef included_sidlf90array_h
#include "sidlf90array.h"
#endif
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_Exception_h
#include "sidl_Exception.h"
#endif
#include <stdio.h>
#include "sidl_rmi_TicketBook_IOR.h"
#include "sidl_rmi_TicketBook_fAbbrev.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_RuntimeException_IOR.h"
#include "sidl_rmi_Response_IOR.h"
#include "sidl_rmi_Ticket_IOR.h"
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
/*
 * Includes for all method dependencies.
 */

#ifndef included_sidl_BaseInterface_fStub_h
#include "sidl_BaseInterface_fStub.h"
#endif
#ifndef included_sidl_ClassInfo_fStub_h
#include "sidl_ClassInfo_fStub.h"
#endif
#ifndef included_sidl_RuntimeException_fStub_h
#include "sidl_RuntimeException_fStub.h"
#endif
#ifndef included_sidl_rmi_Response_fStub_h
#include "sidl_rmi_Response_fStub.h"
#endif
#ifndef included_sidl_rmi_Ticket_fStub_h
#include "sidl_rmi_Ticket_fStub.h"
#endif
#ifndef included_sidl_rmi_TicketBook_fStub_h
#include "sidl_rmi_TicketBook_fStub.h"
#endif

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct sidl_rmi_TicketBook__object* 
  sidl_rmi_TicketBook__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct sidl_rmi_TicketBook__object* 
  sidl_rmi_TicketBook__IHConnect(struct sidl_rmi_InstanceHandle__object 
  *instance, struct sidl_BaseInterface__object **_ex);
/*
 * Remote Connector for the class.
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_rconnect_m,SIDL_RMI_TICKETBOOK_RCONNECT_M,sidl_rmi_TicketBook_rConnect_m)
(
  int64_t *self,
  SIDL_F90_String url
  SIDL_F90_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F90_STR(url),
      SIDL_F90_STR_LEN(url));
  _proxy_self = sidl_rmi_TicketBook__remoteConnect(_proxy_url, 1,
    &_proxy_exception);
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *self = (ptrdiff_t)_proxy_self;
  }
  free((void *)_proxy_url);
}
/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__cast_m,SIDL_RMI_TICKETBOOK__CAST_M,sidl_rmi_TicketBook__cast_m)
(
  int64_t *ref,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_BaseInterface__object  *_base =
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*ref;
  struct sidl_BaseInterface__object *proxy_exception;

  *retval = 0;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.TicketBook",
      (void*)sidl_rmi_TicketBook__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "sidl.rmi.TicketBook", &proxy_exception);
  } else {
    *retval = 0;
    proxy_exception = 0;
  }
  EXIT:
  *exception = (ptrdiff_t)proxy_exception;
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__cast2_m,SIDL_RMI_TICKETBOOK__CAST2_M,sidl_rmi_TicketBook__cast2_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F90_STR(name),
      SIDL_F90_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self->d_object,
      _proxy_name,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
  free((void *)_proxy_name);
}


/*
 * Select and execute a method by name
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__exec_m,SIDL_RMI_TICKETBOOK__EXEC_M,sidl_rmi_TicketBook__exec_m)
(
  int64_t *self,
  SIDL_F90_String methodName
  SIDL_F90_STR_NEAR_LEN_DECL(methodName),
  int64_t *inArgs,
  int64_t *outArgs,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(methodName)
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _proxy_methodName =
    sidl_copy_fortran_str(SIDL_F90_STR(methodName),
      SIDL_F90_STR_LEN(methodName));
  _proxy_inArgs =
    (struct sidl_rmi_Call__object*)
    (ptrdiff_t)(*inArgs);
  _proxy_outArgs =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*outArgs);
  _epv = _proxy_self->d_epv;
  (*(_epv->f__exec))(
    _proxy_self->d_object,
    _proxy_methodName,
    _proxy_inArgs,
    _proxy_outArgs,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_methodName);
}


/*
 * Get the URL of the Implementation of this object (for RMI)
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__geturl_m,SIDL_RMI_TICKETBOOK__GETURL_M,sidl_rmi_TicketBook__getURL_m)
(
  int64_t *self,
  SIDL_F90_String retval
  SIDL_F90_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__getURL))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    sidl_copy_c_str(
      SIDL_F90_STR(retval),
      SIDL_F90_STR_LEN(retval),
      _proxy_retval);
  }
  free((void *)_proxy_retval);
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__isremote_m,SIDL_RMI_TICKETBOOK__ISREMOTE_M,sidl_rmi_TicketBook__isRemote_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__isRemote))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__islocal_m,SIDL_RMI_TICKETBOOK__ISLOCAL_M,sidl_rmi_TicketBook__isLocal_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    !(*(_epv->f__isRemote))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}


/*
 * Method to set whether or not method hooks should be invoked.
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__set_hooks_m,SIDL_RMI_TICKETBOOK__SET_HOOKS_M,sidl_rmi_TicketBook__set_hooks_m)
(
  int64_t *self,
  SIDL_F90_Bool *on,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _proxy_on = ((*on == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f__set_hooks))(
    _proxy_self->d_object,
    _proxy_on,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
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

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_addref_m,SIDL_RMI_TICKETBOOK_ADDREF_M,sidl_rmi_TicketBook_addRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self->d_object,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_deleteref_m,SIDL_RMI_TICKETBOOK_DELETEREF_M,sidl_rmi_TicketBook_deleteRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self->d_object,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_issame_m,SIDL_RMI_TICKETBOOK_ISSAME_M,sidl_rmi_TicketBook_isSame_m)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self->d_object,
      _proxy_iobj,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_istype_m,SIDL_RMI_TICKETBOOK_ISTYPE_M,sidl_rmi_TicketBook_isType_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  SIDL_F90_Bool *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F90_STR(name),
      SIDL_F90_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self->d_object,
      _proxy_name,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_getclassinfo_m,SIDL_RMI_TICKETBOOK_GETCLASSINFO_M,sidl_rmi_TicketBook_getClassInfo_m)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
}

/*
 *  blocks until the Response is recieved 
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_block_m,SIDL_RMI_TICKETBOOK_BLOCK_M,sidl_rmi_TicketBook_block_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_block))(
    _proxy_self->d_object,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 *  
 * returns immediately: true iff the Response is already
 * received 
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_test_m,SIDL_RMI_TICKETBOOK_TEST_M,sidl_rmi_TicketBook_test_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_test))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}

/*
 *  creates an empty container specialized for Tickets 
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_createemptyticketbook_m,SIDL_RMI_TICKETBOOK_CREATEEMPTYTICKETBOOK_M,sidl_rmi_TicketBook_createEmptyTicketBook_m)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_createEmptyTicketBook))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
}

/*
 *  returns immediately: returns Response or null 
 * (NOTE: needed for implementors of communication
 * libraries, not expected for general use).
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_getresponse_m,SIDL_RMI_TICKETBOOK_GETRESPONSE_M,sidl_rmi_TicketBook_getResponse_m)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_rmi_Response__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getResponse))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
}

/*
 *  insert a ticket with a user-specified ID 
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_insertwithid_m,SIDL_RMI_TICKETBOOK_INSERTWITHID_M,sidl_rmi_TicketBook_insertWithID_m)
(
  int64_t *self,
  int64_t *t,
  int32_t *id,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_rmi_Ticket__object* _proxy_t = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _proxy_t =
    (struct sidl_rmi_Ticket__object*)
    (ptrdiff_t)(*t);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_insertWithID))(
    _proxy_self->d_object,
    _proxy_t,
    *id,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 *  insert a ticket and issue a unique ID 
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_insert_m,SIDL_RMI_TICKETBOOK_INSERT_M,sidl_rmi_TicketBook_insert_m)
(
  int64_t *self,
  int64_t *t,
  int32_t *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_rmi_Ticket__object* _proxy_t = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _proxy_t =
    (struct sidl_rmi_Ticket__object*)
    (ptrdiff_t)(*t);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_insert))(
      _proxy_self->d_object,
      _proxy_t,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 *  remove a ready ticket from the TicketBook
 * returns 0 (and null) on an empty TicketBook
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_removeready_m,SIDL_RMI_TICKETBOOK_REMOVEREADY_M,sidl_rmi_TicketBook_removeReady_m)
(
  int64_t *self,
  int64_t *t,
  int32_t *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  struct sidl_rmi_Ticket__object* _proxy_t = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_removeReady))(
      _proxy_self->d_object,
      &_proxy_t,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *t = (ptrdiff_t)_proxy_t;
  }
}

/*
 * immediate, returns the number of Tickets in the book.
 */

void
SIDLFortran90Symbol(sidl_rmi_ticketbook_isempty_m,SIDL_RMI_TICKETBOOK_ISEMPTY_M,sidl_rmi_TicketBook_isEmpty_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_TicketBook__epv *_epv = NULL;
  struct sidl_rmi_TicketBook__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_TicketBook__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isEmpty))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_createcol_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_CREATECOL_M,
                  sidl_rmi_TicketBook__array_createCol_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_createrow_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_CREATEROW_M,
                  sidl_rmi_TicketBook__array_createRow_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_create1d_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_CREATE1D_M,
                  sidl_rmi_TicketBook__array_create1d_m)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_create2dcol_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_CREATE2DCOL_M,
                  sidl_rmi_TicketBook__array_create2dCol_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_create2drow_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_CREATE2DROW_M,
                  sidl_rmi_TicketBook__array_create2dRow_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_addref_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_ADDREF_M,
                  sidl_rmi_TicketBook__array_addRef_m)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_deleteref_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_DELETEREF_M,
                  sidl_rmi_TicketBook__array_deleteRef_m)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_get1_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_GET1_M,
                  sidl_rmi_TicketBook__array_get1_m)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get1((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_get2_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_GET2_M,
                  sidl_rmi_TicketBook__array_get2_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get2((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_get3_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_GET3_M,
                  sidl_rmi_TicketBook__array_get3_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get3((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_get4_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_GET4_M,
                  sidl_rmi_TicketBook__array_get4_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get4((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_get5_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_GET5_M,
                  sidl_rmi_TicketBook__array_get5_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get5((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_get6_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_GET6_M,
                  sidl_rmi_TicketBook__array_get6_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get6((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_get7_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_GET7_M,
                  sidl_rmi_TicketBook__array_get7_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int32_t *i7, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get7((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6, *i7);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_get_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_GET_M,
                  sidl_rmi_TicketBook__array_get_m)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_set1_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SET1_M,
                  sidl_rmi_TicketBook__array_set1_m)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_set2_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SET2_M,
                  sidl_rmi_TicketBook__array_set2_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_set3_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SET3_M,
                  sidl_rmi_TicketBook__array_set3_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_set4_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SET4_M,
                  sidl_rmi_TicketBook__array_set4_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_set5_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SET5_M,
                  sidl_rmi_TicketBook__array_set5_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int64_t *value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_set6_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SET6_M,
                  sidl_rmi_TicketBook__array_set6_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int64_t *value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_set7_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SET7_M,
                  sidl_rmi_TicketBook__array_set7_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int32_t *i7,
   int64_t *value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, *i7,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_set_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SET_M,
                  sidl_rmi_TicketBook__array_set_m)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_dimen_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_DIMEN_M,
                  sidl_rmi_TicketBook__array_dimen_m)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_lower_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_LOWER_M,
                  sidl_rmi_TicketBook__array_lower_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_upper_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_UPPER_M,
                  sidl_rmi_TicketBook__array_upper_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_length_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_LENGTH_M,
                  sidl_rmi_TicketBook__array_length_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_stride_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_STRIDE_M,
                  sidl_rmi_TicketBook__array_stride_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_iscolumnorder_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_ISCOLUMNORDER_M,
                  sidl_rmi_TicketBook__array_isColumnOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_isroworder_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_ISROWORDER_M,
                  sidl_rmi_TicketBook__array_isRowOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_copy_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_COPY_M,
                  sidl_rmi_TicketBook__array_copy_m)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_smartcopy_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SMARTCOPY_M,
                  sidl_rmi_TicketBook__array_smartCopy_m)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_slice_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_SLICE_M,
                  sidl_rmi_TicketBook__array_slice_m)
  (int64_t *src,
   int32_t *dimen,
   int32_t numElem[],
   int32_t srcStart[],
   int32_t srcStride[],
   int32_t newStart[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_slice((struct sidl_interface__array *)(ptrdiff_t)*src,
      *dimen, numElem, srcStart, srcStride, newStart);
}

void
SIDLFortran90Symbol(sidl_rmi_ticketbook__array_ensure_m,
                  SIDL_RMI_TICKETBOOK__ARRAY_ENSURE_M,
                  sidl_rmi_TicketBook__array_ensure_m)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_ensure((struct sidl_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

#include <stdlib.h>
#include <string.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#include "sidl_Exception.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidl_rmi__TicketBook__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_rmi__TicketBook__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_rmi__TicketBook__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_rmi__TicketBook__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 10;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct sidl_rmi__TicketBook__epv s_rem_epv__sidl_rmi__ticketbook;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

static struct sidl_rmi_Ticket__epv s_rem_epv__sidl_rmi_ticket;

static struct sidl_rmi_TicketBook__epv s_rem_epv__sidl_rmi_ticketbook;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidl_rmi__TicketBook__cast(
  struct sidl_rmi__TicketBook__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.rmi.TicketBook");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_sidl_rmi_ticketbook);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.rmi.Ticket");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_rmi_ticket);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.BaseInterface");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_baseinterface);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.rmi._TicketBook");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*,
      struct sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct sidl_rmi__TicketBook__remote*)self->d_data)->d_ih,
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidl_rmi__TicketBook__delete(
  struct sidl_rmi__TicketBook__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidl_rmi__TicketBook__getURL(
  struct sidl_rmi__TicketBook__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_sidl_rmi__TicketBook__raddRef(
  struct sidl_rmi__TicketBook__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
  sidl_rmi_Response _rsvp = NULL;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "addRef", _ex ); SIDL_CHECK(*_ex);
  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
  /* Check for exceptions */
  netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
  if(netex != NULL) {
    sidl_BaseInterface throwaway_exception = NULL;
    *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(netex,
      &throwaway_exception);
    return;
  }

  /* cleanup and return */
  EXIT:
  if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
  if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
  return;
}

/* REMOTE ISREMOTE: returns true if this object is Remote (it is). */
static sidl_bool
remote_sidl_rmi__TicketBook__isRemote(
    struct sidl_rmi__TicketBook__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_sidl_rmi__TicketBook__set_hooks(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* in */ sidl_bool on,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook._set_hooks.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_sidl_rmi__TicketBook__exec(
  struct sidl_rmi__TicketBook__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidl_rmi__TicketBook_addRef(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi__TicketBook__remote* r_obj = (struct 
      sidl_rmi__TicketBook__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidl_rmi__TicketBook_deleteRef(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi__TicketBook__remote* r_obj = (struct 
      sidl_rmi__TicketBook__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount--;
    if(r_obj->d_refcount == 0) {
      sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
      free(r_obj);
      free(self);
    }
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_sidl_rmi__TicketBook_isSame(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* in */ struct sidl_BaseInterface__object* iobj,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isSame", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(iobj){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.isSame.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_sidl_rmi__TicketBook_isType(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* in */ const char* name,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.isType.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_sidl_rmi__TicketBook_getClassInfo(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_ClassInfo__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.getClassInfo.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_ClassInfo__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:block */
static void
remote_sidl_rmi__TicketBook_block(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "block", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.block.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:test */
static sidl_bool
remote_sidl_rmi__TicketBook_test(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "test", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.test.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:createEmptyTicketBook */
static struct sidl_rmi_TicketBook__object*
remote_sidl_rmi__TicketBook_createEmptyTicketBook(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_rmi_TicketBook__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "createEmptyTicketBook", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.createEmptyTicketBook.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_rmi_TicketBook__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getResponse */
static struct sidl_rmi_Response__object*
remote_sidl_rmi__TicketBook_getResponse(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_rmi_Response__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getResponse", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.getResponse.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_rmi_Response__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:insertWithID */
static void
remote_sidl_rmi__TicketBook_insertWithID(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* in */ struct sidl_rmi_Ticket__object* t,
  /* in */ int32_t id,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "insertWithID", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(t){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)t,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "t", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "t", NULL, _ex);SIDL_CHECK(*_ex);
    }
    sidl_rmi_Invocation_packInt( _inv, "id", id, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.insertWithID.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:insert */
static int32_t
remote_sidl_rmi__TicketBook_insert(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* in */ struct sidl_rmi_Ticket__object* t,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "insert", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(t){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)t,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "t", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "t", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.insert.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:removeReady */
static int32_t
remote_sidl_rmi__TicketBook_removeReady(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* out */ struct sidl_rmi_Ticket__object** t,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* t_str= NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "removeReady", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.removeReady.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackString( _rsvp, "t", &t_str, _ex);SIDL_CHECK(*_ex);
    *t = sidl_rmi_Ticket__connectI(t_str, FALSE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isEmpty */
static sidl_bool
remote_sidl_rmi__TicketBook_isEmpty(
  /* in */ struct sidl_rmi__TicketBook__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__TicketBook__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isEmpty", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._TicketBook.isEmpty.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void sidl_rmi__TicketBook__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidl_rmi__TicketBook__epv* epv = &s_rem_epv__sidl_rmi__ticketbook;
  struct sidl_BaseInterface__epv*   e0  = &s_rem_epv__sidl_baseinterface;
  struct sidl_rmi_Ticket__epv*      e1  = &s_rem_epv__sidl_rmi_ticket;
  struct sidl_rmi_TicketBook__epv*  e2  = &s_rem_epv__sidl_rmi_ticketbook;

  epv->f__cast                      = remote_sidl_rmi__TicketBook__cast;
  epv->f__delete                    = remote_sidl_rmi__TicketBook__delete;
  epv->f__exec                      = remote_sidl_rmi__TicketBook__exec;
  epv->f__getURL                    = remote_sidl_rmi__TicketBook__getURL;
  epv->f__raddRef                   = remote_sidl_rmi__TicketBook__raddRef;
  epv->f__isRemote                  = remote_sidl_rmi__TicketBook__isRemote;
  epv->f__set_hooks                 = remote_sidl_rmi__TicketBook__set_hooks;
  epv->f__ctor                      = NULL;
  epv->f__ctor2                     = NULL;
  epv->f__dtor                      = NULL;
  epv->f_addRef                     = remote_sidl_rmi__TicketBook_addRef;
  epv->f_deleteRef                  = remote_sidl_rmi__TicketBook_deleteRef;
  epv->f_isSame                     = remote_sidl_rmi__TicketBook_isSame;
  epv->f_isType                     = remote_sidl_rmi__TicketBook_isType;
  epv->f_getClassInfo               = remote_sidl_rmi__TicketBook_getClassInfo;
  epv->f_block                      = remote_sidl_rmi__TicketBook_block;
  epv->f_test                       = remote_sidl_rmi__TicketBook_test;
  epv->f_createEmptyTicketBook      = 
    remote_sidl_rmi__TicketBook_createEmptyTicketBook;
  epv->f_getResponse                = remote_sidl_rmi__TicketBook_getResponse;
  epv->f_insertWithID               = remote_sidl_rmi__TicketBook_insertWithID;
  epv->f_insert                     = remote_sidl_rmi__TicketBook_insert;
  epv->f_removeReady                = remote_sidl_rmi__TicketBook_removeReady;
  epv->f_isEmpty                    = remote_sidl_rmi__TicketBook_isEmpty;

  e0->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e0->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e0->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e0->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast                 = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete               = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL               = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef              = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote             = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks            = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec                 = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef                = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame                = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType                = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo          = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e1->f_block                 = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_block;
  e1->f_test                  = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_test;
  e1->f_createEmptyTicketBook = (struct sidl_rmi_TicketBook__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_createEmptyTicketBook;
  e1->f_getResponse           = (struct sidl_rmi_Response__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getResponse;

  e2->f__cast                 = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete               = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL               = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef              = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote             = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks            = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec                 = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_addRef                = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame                = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType                = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo          = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e2->f_block                 = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_block;
  e2->f_test                  = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_test;
  e2->f_createEmptyTicketBook = (struct sidl_rmi_TicketBook__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_createEmptyTicketBook;
  e2->f_getResponse           = (struct sidl_rmi_Response__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getResponse;
  e2->f_insertWithID          = (void (*)(void*,struct sidl_rmi_Ticket__object*,
    int32_t,struct sidl_BaseInterface__object **)) epv->f_insertWithID;
  e2->f_insert                = (int32_t (*)(void*,
    struct sidl_rmi_Ticket__object*,
    struct sidl_BaseInterface__object **)) epv->f_insert;
  e2->f_removeReady           = (int32_t (*)(void*,
    struct sidl_rmi_Ticket__object**,
    struct sidl_BaseInterface__object **)) epv->f_removeReady;
  e2->f_isEmpty               = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_isEmpty;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidl_rmi_TicketBook__object*
sidl_rmi_TicketBook__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi__TicketBook__object* self;

  struct sidl_rmi__TicketBook__object* s0;

  struct sidl_rmi__TicketBook__remote* r_obj;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    sidl_BaseInterface bi = 
      (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
      objectID, _ex);
    if(ar) {
      sidl_BaseInterface_addRef(bi, _ex);
    }
    return sidl_rmi_TicketBook__rmicast(bi, _ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_rmi__TicketBook__object*) malloc(
      sizeof(struct sidl_rmi__TicketBook__object));

  r_obj =
    (struct sidl_rmi__TicketBook__remote*) malloc(
      sizeof(struct sidl_rmi__TicketBook__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi__TicketBook__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_rmi_ticket.d_epv    = &s_rem_epv__sidl_rmi_ticket;
  s0->d_sidl_rmi_ticket.d_object = (void*) self;

  s0->d_sidl_rmi_ticketbook.d_epv    = &s_rem_epv__sidl_rmi_ticketbook;
  s0->d_sidl_rmi_ticketbook.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi__ticketbook;

  self->d_data = (void*) r_obj;

  return sidl_rmi_TicketBook__rmicast(self, _ex);
}
/* Create an instance that uses an already existing  */
/* InstanceHandel to connect to an existing remote object. */
static struct sidl_rmi_TicketBook__object*
sidl_rmi_TicketBook__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi__TicketBook__object* self;

  struct sidl_rmi__TicketBook__object* s0;

  struct sidl_rmi__TicketBook__remote* r_obj;
  self =
    (struct sidl_rmi__TicketBook__object*) malloc(
      sizeof(struct sidl_rmi__TicketBook__object));

  r_obj =
    (struct sidl_rmi__TicketBook__remote*) malloc(
      sizeof(struct sidl_rmi__TicketBook__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi__TicketBook__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_rmi_ticket.d_epv    = &s_rem_epv__sidl_rmi_ticket;
  s0->d_sidl_rmi_ticket.d_object = (void*) self;

  s0->d_sidl_rmi_ticketbook.d_epv    = &s_rem_epv__sidl_rmi_ticketbook;
  s0->d_sidl_rmi_ticketbook.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi__ticketbook;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance, _ex);
  return sidl_rmi_TicketBook__rmicast(self, _ex);
}
/*
 * Cast method for interface and class type conversions.
 */

struct sidl_rmi_TicketBook__object*
sidl_rmi_TicketBook__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct sidl_rmi_TicketBook__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.TicketBook",
      (void*)sidl_rmi_TicketBook__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct sidl_rmi_TicketBook__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.rmi.TicketBook", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct sidl_rmi_TicketBook__object*
sidl_rmi_TicketBook__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return sidl_rmi_TicketBook__remoteConnect(url, ar, _ex);
}

