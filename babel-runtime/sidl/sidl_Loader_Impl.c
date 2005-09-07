/*
 * File:          sidl_Loader_Impl.c
 * Symbol:        sidl.Loader-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.Loader
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
 * babel-version = 0.10.10
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.Loader" (version 0.9.3)
 * 
 * Class <code>Loader</code> manages dyanamic loading and symbol name
 * resolution for the sidl runtime system.  The <code>Loader</code> class
 * manages a library search path and keeps a record of all libraries
 * loaded through this interface, including the initial "global" symbols
 * in the main program.
 * 
 * Unless explicitly set, the <code>Loader</code> uses the default
 * <code>sidl.Finder</code> implemented in <code>sidl.DFinder</code>.
 * This class searches the filesystem for <code>.scl</code> files when
 * trying to find a class. The initial path is taken from the
 * environment variable SIDL_DLL_PATH, which is a semi-colon
 * separated sequence of URIs as described in class <code>DLL</code>.
 */

#include "sidl_Loader_Impl.h"

#line 62 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidl.Loader._includes) */
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
#include "sidl_Finder.h"
#include "sidl_DFinder.h"

#ifndef NULL
#define NULL 0
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

static const char * const s_URI_HEADERS[] = { 
  /* sorted list of URI protocols understood */
  "file:",
  "ftp:",
  "http:",
  "lib:",
  "main:"
};

/*
 * Static data members used by sidl.Loader
 */
typedef struct DLL_List {
   sidl_DLL d_dll;
   struct DLL_List* d_next;
} DLL_List;

static DLL_List* s_dll_list    = NULL;
static sidl_Finder s_finder   = NULL; 

/*
 * Initialize the list of DLLs if it has not yet been initialized.  The initial
 * DLL list contains only the single DLL library "main:".
 */
static void initialize_dll_list(void)
{
  if (!s_dll_list) {
    sidl_DLL dll = sidl_DLL__create();
    if (sidl_DLL_loadLibrary(dll, "main:", TRUE, FALSE)) {
      DLL_List* item = (DLL_List*) malloc(sizeof(DLL_List));
      item->d_dll = dll;
      item->d_next = NULL;
      s_dll_list = item;
    } else {
      sidl_DLL_deleteRef(dll);
    }
  }
}

/* DO-NOT-DELETE splicer.end(sidl.Loader._includes) */
#line 127 "sidl_Loader_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_Loader__load(
  void)
{
#line 141 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader._load) */
  /*Use our friend the static initilizer to make sure there's always a Finder*/
  sidl_DFinder temp_find = NULL; 
  temp_find = sidl_DFinder__create(); 
  s_finder = sidl_Finder__cast(temp_find); 
  /* DO-NOT-DELETE splicer.end(sidl.Loader._load) */
#line 150 "sidl_Loader_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_Loader__ctor(
  /* in */ sidl_Loader self)
{
#line 162 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader._ctor) */

  /*
   * All methods in this class are static, so there is nothing to be
   * done in the constructor.
   */

  /* DO-NOT-DELETE splicer.end(sidl.Loader._ctor) */
#line 175 "sidl_Loader_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_Loader__dtor(
  /* in */ sidl_Loader self)
{
#line 186 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader._dtor) */

  /*
   * All methods in this class are static, so there is nothing to be
   * done in the destructor.
   */

  /* DO-NOT-DELETE splicer.end(sidl.Loader._dtor) */
#line 201 "sidl_Loader_Impl.c"
}

/*
 * Load the specified library if it has not already been loaded.
 * The URI format is defined in class <code>DLL</code>.  The search
 * path is not searched to resolve the library name.
 * 
 * @param uri          the URI to load. This can be a .la file
 *                     (a metadata file produced by libtool) or
 *                     a shared library binary (i.e., .so,
 *                     .dll or whatever is appropriate for your
 *                     OS)
 * @param loadGlobally <code>true</code> means that the shared
 *                     library symbols will be loaded into the
 *                     global namespace; <code>false</code> 
 *                     means they will be loaded into a 
 *                     private namespace. Some operating systems
 *                     may not be able to honor the value presented
 *                     here.
 * @param loadLazy     <code>true</code> instructs the loader to
 *                     that symbols can be resolved as needed (lazy)
 *                     instead of requiring everything to be resolved
 *                     now.
 * @return if the load was successful, a non-NULL DLL object is returned.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_loadLibrary"

#ifdef __cplusplus
extern "C"
#endif
sidl_DLL
impl_sidl_Loader_loadLibrary(
  /* in */ const char* uri,
  /* in */ sidl_bool loadGlobally,
  /* in */ sidl_bool loadLazy)
{
#line 232 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.loadLibrary) */
  int ok = FALSE;
  sidl_DLL dll = sidl_DLL__create();

  ok = sidl_DLL_loadLibrary(dll, uri, loadGlobally, loadLazy);

  if (ok) {
    impl_sidl_Loader_addDLL(dll);
    return dll;
  }
  else {
    sidl_DLL_deleteRef(dll);
    return NULL;
  }
  /* DO-NOT-DELETE splicer.end(sidl.Loader.loadLibrary) */
#line 256 "sidl_Loader_Impl.c"
}

/*
 * Append the specified DLL to the beginning of the list of already
 * loaded DLLs.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_addDLL"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_Loader_addDLL(
  /* in */ sidl_DLL dll)
{
#line 264 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.addDLL) */
  if (dll) {
    DLL_List* item = NULL;
    initialize_dll_list();
    item = (DLL_List*) malloc(sizeof(DLL_List));
    sidl_DLL_addRef(dll);
    item->d_dll = dll;
    item->d_next = s_dll_list;
    s_dll_list = item;
  }
  /* DO-NOT-DELETE splicer.end(sidl.Loader.addDLL) */
#line 286 "sidl_Loader_Impl.c"
}

/*
 * Unload all dynamic link libraries.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_unloadLibraries"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_Loader_unloadLibraries(
  void)
{
#line 294 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.unloadLibraries) */
  DLL_List* head = s_dll_list;

  while (head) {
    DLL_List* next = head->d_next;
    sidl_DLL_deleteRef(head->d_dll);
    free(head);
    head = next;
  }

  s_dll_list = NULL;
  /* DO-NOT-DELETE splicer.end(sidl.Loader.unloadLibraries) */
#line 319 "sidl_Loader_Impl.c"
}

/*
 * Find a DLL containing the specified information for a sidl
 * class. This method searches SCL files in the search path looking
 * for a shared library that contains the client-side or IOR
 * for a particular sidl class.
 * 
 * This call is implemented by calling the current
 * <code>Finder</code>. The default finder searches the local
 * file system for <code>.scl</code> files to locate the
 * target class/interface.
 * 
 * @param sidl_name  the fully qualified (long) name of the
 *                   class/interface to be found. Package names
 *                   are separated by period characters from each
 *                   other and the class/interface name.
 * @param target     to find a client-side binding, this is
 *                   normally the name of the language.
 *                   To find the implementation of a class
 *                   in order to make one, you should pass
 *                   the string "ior/impl" here.
 * @param lScope     this specifies whether the symbols should
 *                   be loaded into the global scope, a local
 *                   scope, or use the setting in the SCL file.
 * @param lResolve   this specifies whether symbols should be
 *                   resolved as needed (LAZY), completely
 *                   resolved at load time (NOW), or use the
 *                   setting from the SCL file.
 * @return a non-NULL object means the search was successful.
 *         The DLL has already been added.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_findLibrary"

#ifdef __cplusplus
extern "C"
#endif
sidl_DLL
impl_sidl_Loader_findLibrary(
  /* in */ const char* sidl_name,
  /* in */ const char* target,
  /* in */ enum sidl_Scope__enum lScope,
  /* in */ enum sidl_Resolve__enum lResolve)
{
#line 352 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.findLibrary) */
  return sidl_Finder_findLibrary(s_finder, sidl_name, target, lScope, lResolve);
  /* DO-NOT-DELETE splicer.end(sidl.Loader.findLibrary) */
#line 370 "sidl_Loader_Impl.c"
}

/*
 * Set the search path, which is a semi-colon separated sequence of
 * URIs as described in class <code>DLL</code>.  This method will
 * invalidate any existing search path.
 * 
 * This updates the search path in the current <code>Finder</code>.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_setSearchPath"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_Loader_setSearchPath(
  /* in */ const char* path_name)
{
#line 375 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.setSearchPath) */
  sidl_Finder_setSearchPath(s_finder, path_name);
  /* DO-NOT-DELETE splicer.end(sidl.Loader.setSearchPath) */
#line 395 "sidl_Loader_Impl.c"
}

/*
 * Return the current search path.  The default
 * <code>Finder</code> initializes the search path
 * from environment variable SIDL_DLL_PATH.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_getSearchPath"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_Loader_getSearchPath(
  void)
{
#line 397 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.getSearchPath) */
  return sidl_Finder_getSearchPath(s_finder);
  /* DO-NOT-DELETE splicer.end(sidl.Loader.getSearchPath) */
#line 419 "sidl_Loader_Impl.c"
}

/*
 * Append the specified path fragment to the beginning of the
 * current search path.  This method operates on the Loader's
 * current <code>Finder</code>. This will add a path to the
 * current search path. Normally, the search path is initialized
 * from the SIDL_DLL_PATH environment variable.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_addSearchPath"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_Loader_addSearchPath(
  /* in */ const char* path_fragment)
{
#line 420 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.addSearchPath) */
  sidl_Finder_addSearchPath(s_finder, path_fragment);
  /* DO-NOT-DELETE splicer.end(sidl.Loader.addSearchPath) */
#line 444 "sidl_Loader_Impl.c"
}

/*
 * This method sets the <code>Finder</code> that
 * <code>Loader</code> will use to find DLLs.  If no
 * <code>Finder</code> is set or if NULL is passed in, the Default
 * Finder <code>DFinder</code> will be used.
 * 
 * Future calls to <code>findLibrary</code>,
 * <code>addSearchPath</code>, <code>getSearchPath</code>, and
 * <code>setSearchPath</code> are deligated to the
 * <code>Finder</code> set here.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_setFinder"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_Loader_setFinder(
  /* in */ sidl_Finder f)
{
#line 447 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.setFinder) */
  sidl_DFinder temp_find = NULL;
  if(f != NULL)
    s_finder = f;
  else {
    temp_find = sidl_DFinder__create();
    s_finder = sidl_Finder__cast(temp_find);
  } 
  /* DO-NOT-DELETE splicer.end(sidl.Loader.setFinder) */
#line 479 "sidl_Loader_Impl.c"
}

/*
 * This method gets the <code>Finder</code> that <code>Loader</code>
 * uses to find DLLs.  
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_getFinder"

#ifdef __cplusplus
extern "C"
#endif
sidl_Finder
impl_sidl_Loader_getFinder(
  void)
{
#line 473 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.getFinder) */
  sidl_Finder_addRef(s_finder);
  return s_finder;
  /* DO-NOT-DELETE splicer.end(sidl.Loader.getFinder) */
#line 502 "sidl_Loader_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_Finder__object* impl_sidl_Loader_fconnect_sidl_Finder(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_Finder__connect(url, _ex);
}
char * impl_sidl_Loader_fgetURL_sidl_Finder(struct sidl_Finder__object* obj) {
  return sidl_Finder__getURL(obj);
}
struct sidl_ClassInfo__object* impl_sidl_Loader_fconnect_sidl_ClassInfo(char* 
  url, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidl_Loader_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* 
  obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_DLL__object* impl_sidl_Loader_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_DLL__connect(url, _ex);
}
char * impl_sidl_Loader_fgetURL_sidl_DLL(struct sidl_DLL__object* obj) {
  return sidl_DLL__getURL(obj);
}
struct sidl_Loader__object* impl_sidl_Loader_fconnect_sidl_Loader(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_Loader__connect(url, _ex);
}
char * impl_sidl_Loader_fgetURL_sidl_Loader(struct sidl_Loader__object* obj) {
  return sidl_Loader__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidl_Loader_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidl_Loader_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* impl_sidl_Loader_fconnect_sidl_BaseClass(char* 
  url, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidl_Loader_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* 
  obj) {
  return sidl_BaseClass__getURL(obj);
}
