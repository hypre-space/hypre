/*
 * File:          sidl_rmi_ServerRegistry_IOR.h
 * Symbol:        sidl.rmi.ServerRegistry-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_rmi_ServerRegistry_IOR.h,v 1.1 2006/08/29 23:26:42 painter Exp $
 * Description:   Intermediate Object Representation for sidl.rmi.ServerRegistry
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

#ifndef included_sidl_rmi_ServerRegistry_IOR_h
#define included_sidl_rmi_ServerRegistry_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.rmi.ServerRegistry" (version 0.9.15)
 * 
 *  
 * This singleton class is simply a place to register a
 * ServerInfo interface for general access.  This ServerInfo
 * should give info about the ORB being used to export RMI objects
 * for the current Babel process.
 * 
 * This Registry provides two important functions, a way to get
 * the URL for local object we wish to expose over RMI, and a way
 * to tell if an object passed to this process via RMI is actually
 * a local object.  This abilities are protocol specific, the
 * ServerInfo interface must by implemented by the protocol
 * writer.
 * 
 * THIS CLASS IS NOT DESIGNED FOR CONCURRENT WRITE ACCESS.  (Only
 * one server is assumed per Babel process)
 */

struct sidl_rmi_ServerRegistry__array;
struct sidl_rmi_ServerRegistry__object;
struct sidl_rmi_ServerRegistry__sepv;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_RuntimeException__array;
struct sidl_RuntimeException__object;
struct sidl_io_Serializable__array;
struct sidl_io_Serializable__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;
struct sidl_rmi_ServerInfo__array;
struct sidl_rmi_ServerInfo__object;

/*
 * Declare the static method entry point vector.
 */

struct sidl_rmi_ServerRegistry__sepv {
  /* Implicit builtin methods */
  /* 0 */
  /* 1 */
  /* 2 */
  /* 3 */
  /* 4 */
  /* 5 */
  /* 6 */
  void (*f__set_hooks_static)(
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  /* 8 */
  /* 9 */
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in sidl.rmi.ServerRegistry-v0.9.15 */
  void (*f_registerServer)(
    /* in */ struct sidl_rmi_ServerInfo__object* si,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getServerURL)(
    /* in */ const char* objID,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_isLocalObject)(
    /* in */ const char* url,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_io_Serializable__array* (*f_getExceptions)(
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Declare the method entry point vector.
 */

struct sidl_rmi_ServerRegistry__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_rmi_ServerRegistry__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in sidl.rmi.ServerRegistry-v0.9.15 */
};

/*
 * Define the controls structure.
 */


struct sidl_rmi_ServerRegistry__controls {
  int     use_hooks;
};
/*
 * Define the class object structure.
 */

struct sidl_rmi_ServerRegistry__object {
  struct sidl_BaseClass__object        d_sidl_baseclass;
  struct sidl_rmi_ServerRegistry__epv* d_epv;
  void*                                d_data;
};

struct sidl_rmi_ServerRegistry__external {
  struct sidl_rmi_ServerRegistry__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct sidl_rmi_ServerRegistry__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_rmi_ServerRegistry__external*
sidl_rmi_ServerRegistry__externals(void);

extern struct sidl_rmi_ServerRegistry__object*
sidl_rmi_ServerRegistry__new(void* ddata,
  struct sidl_BaseInterface__object ** _ex);

extern struct sidl_rmi_ServerRegistry__sepv*
sidl_rmi_ServerRegistry__statics(void);

extern void sidl_rmi_ServerRegistry__init(
  struct sidl_rmi_ServerRegistry__object* self, void* ddata,
    struct sidl_BaseInterface__object ** _ex);
extern void sidl_rmi_ServerRegistry__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
    struct sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidl_rmi_ServerRegistry__epv **s_arg_epv__sidl_rmi_serverregistry,
    struct sidl_rmi_ServerRegistry__epv 
    **s_arg_epv_hooks__sidl_rmi_serverregistry);
  extern void sidl_rmi_ServerRegistry__fini(
    struct sidl_rmi_ServerRegistry__object* self,
      struct sidl_BaseInterface__object ** _ex);
  extern void sidl_rmi_ServerRegistry__IOR_version(int32_t *major,
    int32_t *minor);

  struct sidl_BaseClass__object* 
    skel_sidl_rmi_ServerRegistry_fconnect_sidl_BaseClass(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* 
    skel_sidl_rmi_ServerRegistry_fcast_sidl_BaseClass(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_sidl_rmi_ServerRegistry_fconnect_sidl_BaseInterface(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_sidl_rmi_ServerRegistry_fcast_sidl_BaseInterface(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_sidl_rmi_ServerRegistry_fconnect_sidl_ClassInfo(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* 
    skel_sidl_rmi_ServerRegistry_fcast_sidl_ClassInfo(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_sidl_rmi_ServerRegistry_fconnect_sidl_RuntimeException(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_sidl_rmi_ServerRegistry_fcast_sidl_RuntimeException(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_io_Serializable__object* 
    skel_sidl_rmi_ServerRegistry_fconnect_sidl_io_Serializable(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_io_Serializable__object* 
    skel_sidl_rmi_ServerRegistry_fcast_sidl_io_Serializable(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_rmi_ServerInfo__object* 
    skel_sidl_rmi_ServerRegistry_fconnect_sidl_rmi_ServerInfo(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_ServerInfo__object* 
    skel_sidl_rmi_ServerRegistry_fcast_sidl_rmi_ServerInfo(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_rmi_ServerRegistry__object* 
    skel_sidl_rmi_ServerRegistry_fconnect_sidl_rmi_ServerRegistry(const char* 
    url, sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_ServerRegistry__object* 
    skel_sidl_rmi_ServerRegistry_fcast_sidl_rmi_ServerRegistry(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_rmi_ServerRegistry__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

  #ifdef __cplusplus
  }
  #endif
  #endif
