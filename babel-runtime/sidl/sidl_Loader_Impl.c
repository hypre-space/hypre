/*
 * File:          sidl_Loader_Impl.c
 * Symbol:        sidl.Loader-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
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
 * babel-version = 0.9.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.Loader" (version 0.9.0)
 * 
 * Class <code>Loader</code> manages dyanamic loading and symbol name
 * resolution for the sidl runtime system.  The <code>Loader</code> class
 * manages a library search path and keeps a record of all libraries
 * loaded through this interface, including the initial "global" symbols
 * in the main program.  Unless explicitly set, the search path is taken
 * from the environment variable SIDL_DLL_PATH, which is a semi-colon
 * separated sequence of URIs as described in class <code>DLL</code>.
 */

#include "sidl_Loader_Impl.h"

#line 57 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
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

static char*     s_search_path = NULL;
static DLL_List* s_dll_list    = NULL;

/*
 * Initialize the search path if it has not yet been initialized.  The initial
 * search path value is taken from environment variable SIDL_DLL_PATH.
 */
static char *get_search_path(void)
{
  if (!s_search_path) {
    s_search_path = sidl_String_strdup(getenv("SIDL_DLL_PATH"));
    if (!s_search_path) {
      s_search_path = sidl_String_strdup("");
    }
  }
  return s_search_path;
}

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

/*
 * A simple utility routine to calculate the current working directory.
 * A null is returned if we cannot determine the current working directory.
 */
static char* util_getCWD(void)
{
  char*  cwd  = NULL;
  size_t size = 64;

  while (TRUE) {
    char* buf = sidl_String_alloc(size);
    if (getcwd(buf, size) != NULL) {
      cwd = buf;
      break;
    } else if (errno == ERANGE) {
      sidl_String_free(buf);
      size *= 2;
    } else {
      sidl_String_free(buf);
      break;
    }
  }

  return cwd;
}

/*
 * Return whether the specified URI refers to a file.  We assume that any
 * URI that we do not understand points to a file.
 */
static int is_file_uri(const char* uri)
{
  int file = TRUE;

  if (sidl_String_startsWith(uri, "file:")) {
    file = TRUE;
  } else if (sidl_String_equals(uri, "main:")) {
    file = FALSE;
  } else if (sidl_String_startsWith(uri, "lib:")) {
    file = FALSE;
#ifdef HAVE_LIBWWW
  } else if (sidl_String_startsWith(uri, "ftp:")) {
    file = FALSE;
  } else if (sidl_String_startsWith(uri, "http:")) {
    file = FALSE;
#endif
  }

  return file;
}

/*
 * Search through the list of DLLs to find out whether we have already
 * loaded the specified uri.
 */
static int already_loaded(const char* uri)
{
  DLL_List* head = NULL;
  int       same = FALSE;

  initialize_dll_list();

  head = s_dll_list;
  while ((head != NULL) && !same) {
    char* name = sidl_DLL_getName(head->d_dll);
    same = sidl_String_equals(uri, name);
    sidl_String_free(name);
    head = head->d_next;
  }

  return same;
}

/**
 * Find the SCL entry matching a class.
 */
static 
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


static
struct sidl_scl_entry *
findSCLEntry(const char *sidl_name,
             const char *target)
{
  struct sidl_scl_entry *result=NULL;
  int len;
  const char *path = get_search_path(), *next;
  char *buffer = malloc(strlen(path)+1);
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
             

static int
chooseScope(enum sidl_Scope__enum uScope,
            enum sidl_Scope__enum sScope)
{
  return (uScope == sidl_Scope_SCLSCOPE) ?
    (sScope == sidl_Scope_GLOBAL) :
    (uScope == sidl_Scope_GLOBAL);
}

static int
chooseResolve(enum sidl_Resolve__enum uResolve,
              enum sidl_Resolve__enum sResolve)
{
  return (uResolve == sidl_Resolve_SCLRESOLVE) ?
    (sResolve == sidl_Resolve_LAZY) :
    (uResolve == sidl_Resolve_LAZY);
}

static
sidl_DLL
loadLibraryFromSCL(struct sidl_scl_entry *scl,
                   const char *sidl_name,
                   enum sidl_Scope__enum   lScope,
                   enum sidl_Resolve__enum lResolve)
{
  return 
    sidl_Loader_loadLibrary(scl->d_uri,
                            chooseScope(lScope, scl->d_scope),
                            chooseResolve(lResolve, scl->d_resolve));
}
/* DO-NOT-DELETE splicer.end(sidl.Loader._includes) */
#line 284 "sidl_Loader_Impl.c"

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader__ctor"

void
impl_sidl_Loader__ctor(
  /*in*/ sidl_Loader self)
{
#line 295 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader._ctor) */

  /*
   * All methods in this class are static, so there is nothing to be
   * done in the constructor.
   */

  /* DO-NOT-DELETE splicer.end(sidl.Loader._ctor) */
#line 306 "sidl_Loader_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader__dtor"

void
impl_sidl_Loader__dtor(
  /*in*/ sidl_Loader self)
{
#line 316 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader._dtor) */

  /*
   * All methods in this class are static, so there is nothing to be
   * done in the destructor.
   */

  /* DO-NOT-DELETE splicer.end(sidl.Loader._dtor) */
#line 329 "sidl_Loader_Impl.c"
}

/*
 * Set the search path, which is a semi-colon separated sequence of
 * URIs as described in class <code>DLL</code>.  This method will
 * invalidate any existing search path.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_setSearchPath"

void
impl_sidl_Loader_setSearchPath(
  /*in*/ const char* path_name)
{
#line 339 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.setSearchPath) */
  sidl_String_free(s_search_path);
  s_search_path = sidl_String_strdup(path_name);
  if (!s_search_path) {
    s_search_path = sidl_String_strdup("");
  }
  /* DO-NOT-DELETE splicer.end(sidl.Loader.setSearchPath) */
#line 353 "sidl_Loader_Impl.c"
}

/*
 * Return the current search path.  If the search path has not been
 * set, then the search path will be taken from environment variable
 * SIDL_DLL_PATH.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_getSearchPath"

char*
impl_sidl_Loader_getSearchPath(
void)
{
#line 361 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.getSearchPath) */
  return sidl_String_strdup(get_search_path());
  /* DO-NOT-DELETE splicer.end(sidl.Loader.getSearchPath) */
#line 373 "sidl_Loader_Impl.c"
}

/*
 * Append the specified path fragment to the beginning of the
 * current search path.  If the search path has not yet been set
 * by a call to <code>setSearchPath</code>, then this fragment will
 * be appended to the path in environment variable SIDL_DLL_PATH.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_addSearchPath"

void
impl_sidl_Loader_addSearchPath(
  /*in*/ const char* path_fragment)
{
#line 380 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.addSearchPath) */
  if (path_fragment) {
    char* s = sidl_String_concat3(path_fragment, ";", get_search_path());
    sidl_String_free(s_search_path);
    s_search_path = s;
  }
  /* DO-NOT-DELETE splicer.end(sidl.Loader.addSearchPath) */
#line 398 "sidl_Loader_Impl.c"
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

sidl_DLL
impl_sidl_Loader_loadLibrary(
  /*in*/ const char* uri, /*in*/ sidl_bool loadGlobally,
    /*in*/ sidl_bool loadLazy)
{
#line 421 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
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
#line 449 "sidl_Loader_Impl.c"
}

/*
 * Append the specified DLL to the beginning of the list of already
 * loaded DLLs.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_addDLL"

void
impl_sidl_Loader_addDLL(
  /*in*/ sidl_DLL dll)
{
#line 450 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
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
#line 476 "sidl_Loader_Impl.c"
}

/*
 * Unload all dynamic link libraries.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_Loader_unloadLibraries"

void
impl_sidl_Loader_unloadLibraries(
void)
{
#line 477 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
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
#line 506 "sidl_Loader_Impl.c"
}

/*
 * Find a DLL containing the specified information for a sidl
 * class. This method searches SCL files in the search path looking
 * for a shared library that contains the client-side or IOR
 * for a particular sidl class.
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

sidl_DLL
impl_sidl_Loader_findLibrary(
  /*in*/ const char* sidl_name, /*in*/ const char* target,
    /*in*/ enum sidl_Scope__enum lScope,
    /*in*/ enum sidl_Resolve__enum lResolve)
{
#line 526 "../../../babel/runtime/sidl/sidl_Loader_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader.findLibrary) */
  sidl_DLL result = NULL;
  struct sidl_scl_entry *scl = findSCLEntry(sidl_name, target);
  if (scl) {
    result = loadLibraryFromSCL(scl, sidl_name, lScope, lResolve);
    sidl_destroy_scl(scl);
  }
  return result;
  /* DO-NOT-DELETE splicer.end(sidl.Loader.findLibrary) */
#line 554 "sidl_Loader_Impl.c"
}
