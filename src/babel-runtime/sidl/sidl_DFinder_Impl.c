/*
 * File:          sidl_DFinder_Impl.c
 * Symbol:        sidl.DFinder-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name:  $
 * Revision:      @(#) $Id: sidl_DFinder_Impl.c,v 1.5 2006/08/29 22:29:49 painter Exp $
 * Description:   Server-side implementation for sidl.DFinder
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
 * Symbol "sidl.DFinder" (version 0.9.15)
 * 
 * This class is the Default Finder.  If no Finder is set in class Loader,
 * this finder is used.  It uses SCL files from the filesystem to
 * resolve dynamic libraries.
 * 
 * The initial search path is taken from the SIDL_DLL_PATH
 * environment variable.
 */

#include "sidl_DFinder_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidl.DFinder._includes) */
#include "sidl_DLL.h"
#include "sidl_String.h"
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include "sidl_search_scl.h"
#include "sidl_Loader.h"

/*
 * Initialize the search path if it has not yet been initialized.  The initial
 * search path value is taken from environment variable SIDL_DLL_PATH.
 */
char *get_search_path(sidl_DFinder self) /*need self to access data*/
{
  struct sidl_DFinder__data *dptr = sidl_DFinder__get_data(self);
  if (!dptr->d_search_path) {
    dptr->d_search_path = sidl_String_strdup(getenv("SIDL_DLL_PATH"));
    if (!dptr->d_search_path) {
      dptr->d_search_path = sidl_String_strdup("");
    }
  }
  return dptr->d_search_path;
}


/**
 * Find the SCL entry matching a class.
 */

struct sidl_scl_entry *
searchFile(const char            *sidl_name,
           const char            *target,
           const char            *scl_file,
           struct sidl_scl_entry *result)
{
  struct sidl_scl_entry *entry;
  if ((entry = sidl_search_scl(sidl_name, target, scl_file))) {
    if (result) {
      sidl_scl_reportDuplicate(sidl_name, entry, result);
      sidl_destroy_scl(entry);
    }
    else {
      result = entry;
    }
  }
  return result;
}


struct sidl_scl_entry *
findSCLEntry(sidl_DFinder self,
	     const char *sidl_name,
             const char *target)
{
  struct sidl_scl_entry *result=NULL;
  int len;
  const char *path = get_search_path(self), *next;
  char *buffer = (char *)malloc(strlen(path)+1);
  while ((next = strchr(path, ';'))) {
    len = next - path;
    memcpy(buffer, path, len);
    buffer[len] = '\0';
    result = searchFile(sidl_name, target, buffer, result);
    path = next + 1;
  }
  result = searchFile(sidl_name, target, path, result);
  free(buffer);
  return result;
}

int
chooseScope(enum sidl_Scope__enum uScope,
            enum sidl_Scope__enum sScope)
{
  return (uScope == sidl_Scope_SCLSCOPE) ?
    (sScope == sidl_Scope_GLOBAL) :
    (uScope == sidl_Scope_GLOBAL);
}

int
chooseResolve(enum sidl_Resolve__enum uResolve,
              enum sidl_Resolve__enum sResolve)
{
  return (uResolve == sidl_Resolve_SCLRESOLVE) ?
    (sResolve == sidl_Resolve_LAZY) :
    (uResolve == sidl_Resolve_LAZY);
}


sidl_DLL
loadLibraryFromSCL(struct sidl_scl_entry *scl,
                   const char *sidl_name,
                   enum sidl_Scope__enum   lScope,
                   enum sidl_Resolve__enum lResolve,
		   sidl_BaseInterface* _ex)
{
  return 
    sidl_Loader_loadLibrary(scl->d_uri,
                            chooseScope(lScope, scl->d_scope),
                            chooseResolve(lResolve, scl->d_resolve),
			    _ex);
}

/* DO-NOT-DELETE splicer.end(sidl.DFinder._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DFinder__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DFinder__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.DFinder._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidl.DFinder._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DFinder__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DFinder__ctor(
  /* in */ sidl_DFinder self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.DFinder._ctor) */
  struct sidl_DFinder__data *dptr =
    (struct sidl_DFinder__data *)
    malloc(sizeof(struct sidl_DFinder__data));
  if (dptr) {
    dptr->d_search_path = 0;
  }
  sidl_DFinder__set_data(self, dptr);
  /* DO-NOT-DELETE splicer.end(sidl.DFinder._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DFinder__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DFinder__ctor2(
  /* in */ sidl_DFinder self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.DFinder._ctor2) */
  /* Insert-Code-Here {sidl.DFinder._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidl.DFinder._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DFinder__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DFinder__dtor(
  /* in */ sidl_DFinder self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.DFinder._dtor) */
  struct sidl_DFinder__data *dptr =
    sidl_DFinder__get_data(self);
  if (dptr) {
    free((void *)dptr);
    sidl_DFinder__set_data(self, NULL);
  }
  /* DO-NOT-DELETE splicer.end(sidl.DFinder._dtor) */
  }
}

/*
 * Find a DLL containing the specified information for a sidl
 * class. This method searches through the files in set set path
 * looking for a shared library that contains the client-side or IOR
 * for a particular sidl class.
 * 
 * @param sidl_name  the fully qualified (long) name of the
 * class/interface to be found. Package names
 * are separated by period characters from each
 * other and the class/interface name.
 * @param target     to find a client-side binding, this is
 * normally the name of the language.
 * To find the implementation of a class
 * in order to make one, you should pass
 * the string "ior/impl" here.
 * @param lScope     this specifies whether the symbols should
 * be loaded into the global scope, a local
 * scope, or use the setting in the file.
 * @param lResolve   this specifies whether symbols should be
 * resolved as needed (LAZY), completely
 * resolved at load time (NOW), or use the
 * setting from the file.
 * @return a non-NULL object means the search was successful.
 * The DLL has already been added.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DFinder_findLibrary"

#ifdef __cplusplus
extern "C"
#endif
sidl_DLL
impl_sidl_DFinder_findLibrary(
  /* in */ sidl_DFinder self,
  /* in */ const char* sidl_name,
  /* in */ const char* target,
  /* in */ enum sidl_Scope__enum lScope,
  /* in */ enum sidl_Resolve__enum lResolve,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.DFinder.findLibrary) */
  sidl_DLL result = NULL;
  struct sidl_scl_entry *scl = findSCLEntry(self, sidl_name, target);
  if (scl) {
    result = loadLibraryFromSCL(scl, sidl_name, lScope, lResolve, _ex);
    sidl_destroy_scl(scl);
  }
  return result;

  /* DO-NOT-DELETE splicer.end(sidl.DFinder.findLibrary) */
  }
}

/*
 * Set the search path, which is a semi-colon separated sequence of
 * URIs as described in class <code>DLL</code>.  This method will
 * invalidate any existing search path.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DFinder_setSearchPath"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DFinder_setSearchPath(
  /* in */ sidl_DFinder self,
  /* in */ const char* path_name,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.DFinder.setSearchPath) */
  struct sidl_DFinder__data *dptr = sidl_DFinder__get_data(self);
  sidl_String_free(dptr->d_search_path);
  dptr->d_search_path = sidl_String_strdup(path_name);
  if (!dptr->d_search_path) {
    dptr->d_search_path = sidl_String_strdup("");
  }
  /* DO-NOT-DELETE splicer.end(sidl.DFinder.setSearchPath) */
  }
}

/*
 * Return the current search path.  If the search path has not been
 * set, then the search path will be taken from environment variable
 * SIDL_DLL_PATH.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DFinder_getSearchPath"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_DFinder_getSearchPath(
  /* in */ sidl_DFinder self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.DFinder.getSearchPath) */
  return sidl_String_strdup(get_search_path(self));
  /* DO-NOT-DELETE splicer.end(sidl.DFinder.getSearchPath) */
  }
}

/*
 * Append the specified path fragment to the beginning of the
 * current search path.  If the search path has not yet been set
 * by a call to <code>setSearchPath</code>, then this fragment will
 * be appended to the path in environment variable SIDL_DLL_PATH.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DFinder_addSearchPath"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DFinder_addSearchPath(
  /* in */ sidl_DFinder self,
  /* in */ const char* path_fragment,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.DFinder.addSearchPath) */
  struct sidl_DFinder__data *dptr = sidl_DFinder__get_data(self);
  if (path_fragment) {
    char* s = sidl_String_concat3(path_fragment, ";", get_search_path(self));
    sidl_String_free(dptr->d_search_path);
    dptr->d_search_path = s;
  }
  /* DO-NOT-DELETE splicer.end(sidl.DFinder.addSearchPath) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* impl_sidl_DFinder_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_sidl_DFinder_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_DFinder_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_DFinder_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_sidl_DFinder_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_sidl_DFinder_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_DFinder__object* impl_sidl_DFinder_fconnect_sidl_DFinder(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_DFinder__connectI(url, ar, _ex);
}
struct sidl_DFinder__object* impl_sidl_DFinder_fcast_sidl_DFinder(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_DFinder__cast(bi, _ex);
}
struct sidl_DLL__object* impl_sidl_DFinder_fconnect_sidl_DLL(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_DLL__connectI(url, ar, _ex);
}
struct sidl_DLL__object* impl_sidl_DFinder_fcast_sidl_DLL(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_DLL__cast(bi, _ex);
}
struct sidl_Finder__object* impl_sidl_DFinder_fconnect_sidl_Finder(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_Finder__connectI(url, ar, _ex);
}
struct sidl_Finder__object* impl_sidl_DFinder_fcast_sidl_Finder(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_Finder__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_DFinder_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_DFinder_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
