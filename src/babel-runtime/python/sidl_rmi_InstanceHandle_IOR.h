/*
 * File:          sidl_rmi_InstanceHandle_IOR.h
 * Symbol:        sidl.rmi.InstanceHandle-v0.9.15
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Release:       $Name: V1-14-0b $
 * Revision:      @(#) $Id: sidl_rmi_InstanceHandle_IOR.h,v 1.5 2006/08/29 22:29:27 painter Exp $
 * Description:   Intermediate Object Representation for sidl.rmi.InstanceHandle
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

#ifndef included_sidl_rmi_InstanceHandle_IOR_h
#define included_sidl_rmi_InstanceHandle_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.rmi.InstanceHandle" (version 0.9.15)
 * 
 *  
 * This interface holds the state information for handles to
 * remote objects.  Client-side messaging libraries are expected
 * to implement <code>sidl.rmi.InstanceHandle</code>,
 * <code>sidl.rmi.Invocation</code> and
 * <code>sidl.rmi.Response</code>.
 * 
 * Every stub with a connection to a remote object holds a pointer
 * to an InstanceHandle that manages the connection. Multiple
 * stubs may point to the same InstanceHandle, however.  Babel
 * takes care of the reference counting, but the developer should
 * keep concurrency issues in mind.
 * 
 * When a new remote object is created:
 * sidl_rmi_InstanceHandle c = 
 * sidl_rmi_ProtocolFactory_createInstance( url, typeName,
 * _ex );
 * 
 * When a new stub is created to connect to an existing remote
 * instance:
 * sidl_rmi_InstanceHandle c = 
 * sidl_rmi_ProtocolFactory_connectInstance( url, _ex );
 * 
 * When a method is invoked:
 * sidl_rmi_Invocation i = 
 * sidl_rmi_InstanceHandle_createInvocation( methodname );
 * sidl_rmi_Invocation_packDouble( i, "input_val" , 2.0 );
 * sidl_rmi_Invocation_packString( i, "input_str", "Hello" );
 * ...
 * sidl_rmi_Response r = sidl_rmi_Invocation_invokeMethod( i );
 * sidl_rmi_Response_unpackBool( i, "_retval", &succeeded );
 * sidl_rmi_Response_unpackFloat( i, "output_val", &f );
 */

struct sidl_rmi_InstanceHandle__array;
struct sidl_rmi_InstanceHandle__object;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_RuntimeException__array;
struct sidl_RuntimeException__object;
struct sidl_io_Serializable__array;
struct sidl_io_Serializable__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_Invocation__array;
struct sidl_rmi_Invocation__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_rmi_InstanceHandle__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ void* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.rmi.InstanceHandle-v0.9.15 */
  sidl_bool (*f_initCreate)(
    /* in */ void* self,
    /* in */ const char* url,
    /* in */ const char* typeName,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_initConnect)(
    /* in */ void* self,
    /* in */ const char* url,
    /* in */ sidl_bool ar,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_io_Serializable__object* (*f_initUnserialize)(
    /* in */ void* self,
    /* in */ const char* url,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getProtocol)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getObjectID)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getObjectURL)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_rmi_Invocation__object* (*f_createInvocation)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_close)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the interface object structure.
 */

struct sidl_rmi_InstanceHandle__object {
  struct sidl_rmi_InstanceHandle__epv* d_epv;
  void*                                d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
/*
 * Symbol "sidl.rmi._InstanceHandle" (version 1.0)
 */

struct sidl_rmi__InstanceHandle__array;
struct sidl_rmi__InstanceHandle__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_rmi__InstanceHandle__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__delete)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__exec)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f__getURL)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__raddRef)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f__isRemote)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__set_hooks)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__ctor)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__ctor2)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__dtor)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.rmi.InstanceHandle-v0.9.15 */
  sidl_bool (*f_initCreate)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ const char* url,
    /* in */ const char* typeName,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_initConnect)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ const char* url,
    /* in */ sidl_bool ar,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_io_Serializable__object* (*f_initUnserialize)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ const char* url,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getProtocol)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getObjectID)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getObjectURL)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_rmi_Invocation__object* (*f_createInvocation)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* in */ const char* methodName,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_close)(
    /* in */ struct sidl_rmi__InstanceHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.rmi._InstanceHandle-v1.0 */
};

/*
 * Define the class object structure.
 */

struct sidl_rmi__InstanceHandle__object {
  struct sidl_BaseInterface__object      d_sidl_baseinterface;
  struct sidl_rmi_InstanceHandle__object d_sidl_rmi_instancehandle;
  struct sidl_rmi__InstanceHandle__epv*  d_epv;
  void*                                  d_data;
};


struct sidl_rmi__InstanceHandle__remote{
  int d_refcount;
  struct sidl_rmi_InstanceHandle__object *d_ih;
};

#ifdef __cplusplus
}
#endif
#endif
