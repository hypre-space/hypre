/*
 * File:          SIDL_Loader_Impl.c
 * Symbol:        SIDL.Loader-v0.7.5
 * Symbol Type:   class
 * Babel Version: 0.7.5
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for SIDL.Loader
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
 * Symbol "SIDL.Loader" (version 0.7.5)
 * 
 * Class <code>Loader</code> manages dynamic loading and symbol name
 * resolution for the SIDL runtime system.  The <code>Loader</code> class
 * manages a library search path and keeps a record of all libraries
 * loaded through this interface, including the initial "global" symbols
 * in the main program.  Unless explicitly set, the search path is taken
 * from the environment variable SIDL_DLL_PATH, which is a semi-colon
 * separated sequence of URIs as described in class <code>DLL</code>.
 */

#include "SIDL_Loader_Impl.h"

/* DO-NOT-DELETE splicer.begin(SIDL.Loader._includes) */
#include "SIDL_DLL.h"
#include "SIDL_String.h"
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef NULL
#define NULL 0
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

/*
 * Static data members used by SIDL.Loader
 */
typedef struct DLL_List {
   SIDL_DLL d_dll;
   struct DLL_List* d_next;
} DLL_List;

static int       s_search_init = FALSE;
static char*     s_search_path = NULL;
static DLL_List* s_dll_list    = NULL;

/*
 * Initialize the search path if it has not yet been initialized.  The initial
 * search path value is taken from environment variable SIDL_DLL_PATH.
 */
static void initialize_search_path(void)
{
  if (!s_search_init) {
    s_search_path = SIDL_String_strdup(getenv("SIDL_DLL_PATH"));
    s_search_init = TRUE;
  }
}

/*
 * Initialize the list of DLLs if it has not yet been initialized.  The initial
 * DLL list contains only the single DLL library "main:".
 */
static void initialize_dll_list(void)
{
  if (!s_dll_list) {
    SIDL_DLL dll = SIDL_DLL__create();
    if (SIDL_DLL_loadLibrary(dll, "main:")) {
      DLL_List* item = (DLL_List*) malloc(sizeof(DLL_List));
      item->d_dll = dll;
      item->d_next = NULL;
      s_dll_list = item;
    } else {
      SIDL_DLL_deleteReference(dll);
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
    char* buf = SIDL_String_alloc(size);
    if (getcwd(buf, size) != NULL) {
      cwd = buf;
      break;
    } else if (errno == ERANGE) {
      SIDL_String_free(buf);
      size *= 2;
    } else {
      SIDL_String_free(buf);
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

  if (SIDL_String_startsWith(uri, "file:")) {
    file = TRUE;
  } else if (SIDL_String_equals(uri, "main:")) {
    file = FALSE;
  } else if (SIDL_String_startsWith(uri, "lib:")) {
    file = FALSE;
#ifdef HAVE_LIBWWW
  } else if (SIDL_String_startsWith(uri, "ftp:")) {
    file = FALSE;
  } else if (SIDL_String_startsWith(uri, "http:")) {
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
    char* name = SIDL_DLL_getLibraryName(head->d_dll);
    same = SIDL_String_equals(uri, name);
    SIDL_String_free(name);
    head = head->d_next;
  }

  return same;
}

/*
 * Try to load the DLL corresponding to the specified URI and look up
 * the specified name.  If the name cannot be found, then return null.
 */
static void* try_load_dll(const char* uri, const char* linker_name)
{
  void* symbol = NULL;

  if (!already_loaded(uri)) {
    SIDL_DLL dll = SIDL_DLL__create();
    if (SIDL_DLL_loadLibrary(dll, uri)) {
      impl_SIDL_Loader_addDLL(dll);
      symbol = SIDL_DLL_lookupSymbol(dll, linker_name);
    }
    SIDL_DLL_deleteReference(dll);
  }

  return symbol;
}

/*
 * Return whether the specified file is a directory.
 */
static int is_directory(const char* file)
{
   struct stat status;
   return ((!stat(file, &status)) && (S_ISDIR(status.st_mode)));
}

/*
 * Try to find the specified symbol in the specified file URI.  The URI
 * must begin with the "file:" protocol.  If the file is a directory, then
 * recursively search all possible dll files and subdirectories for the
 * match.
 */
static void* find_symbol(const char* uri, const char* linker_name)
{
  void* symbol = NULL;

  if (SIDL_String_startsWith(uri, "file:")) {
    char* file = SIDL_String_substring(uri, 5);

    /*
     * If not a directory, then open the file and check if dll opens OK.
     * If we support LTDL, then allow .la extensions.  If on CYGWIN,
     * then also allow .dll extensions.  Otherwise, allow .so extensions.
     */
    if (!is_directory(file)) {
#ifdef HAVE_LTDL
      if (SIDL_String_endsWith(uri, ".la")) {
        symbol = try_load_dll(uri, linker_name);
      }
#endif
#ifdef __CYGWIN__
      if (SIDL_String_endsWith(uri, ".dll")) {
        symbol = try_load_dll(uri, linker_name);
      }
#else
      if (SIDL_String_endsWith(uri, ".so")) {
        symbol = try_load_dll(uri, linker_name);
      }
#endif

    /*
     * If file is a directory, then recurse through the directory
     */
    } else {
      DIR *dir = opendir(file);
      if (dir) {
        struct dirent* entry = NULL;
        while ((entry = readdir(dir)) != NULL) {
          if (entry->d_name[0] != '.') {
            char* f = SIDL_String_concat4("file:", file, "/", entry->d_name);
            symbol = find_symbol(f, linker_name);
            SIDL_String_free(f);
            if (symbol) {
              break;
            }
          }
        }
        closedir(dir);
      }
    }
    SIDL_String_free(file);
  }

  return symbol;
}
/* DO-NOT-DELETE splicer.end(SIDL.Loader._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader__ctor"

void
impl_SIDL_Loader__ctor(
  SIDL_Loader self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader._ctor) */

  /*
   * All methods in this class are static, so there is nothing to be
   * done in the constructor.
   */

  /* DO-NOT-DELETE splicer.end(SIDL.Loader._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader__dtor"

void
impl_SIDL_Loader__dtor(
  SIDL_Loader self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader._dtor) */

  /*
   * All methods in this class are static, so there is nothing to be
   * done in the destructor.
   */

  /* DO-NOT-DELETE splicer.end(SIDL.Loader._dtor) */
}

/*
 * Set the search path, which is a semi-colon separated sequence of
 * URIs as described in class <code>DLL</code>.  This method will
 * invalidate any existing search path.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader_setSearchPath"

void
impl_SIDL_Loader_setSearchPath(
  const char* path_name)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader.setSearchPath) */
  s_search_init = TRUE;
  SIDL_String_free(s_search_path);
  s_search_path = SIDL_String_strdup(path_name);
  /* DO-NOT-DELETE splicer.end(SIDL.Loader.setSearchPath) */
}

/*
 * Return the current search path.  If the search path has not been
 * set, then the search path will be taken from environment variable
 * SIDL_DLL_PATH.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader_getSearchPath"

char*
impl_SIDL_Loader_getSearchPath(
void)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader.getSearchPath) */
  initialize_search_path();
  return SIDL_String_strdup(s_search_path);
  /* DO-NOT-DELETE splicer.end(SIDL.Loader.getSearchPath) */
}

/*
 * Append the specified path fragment to the beginning of the
 * current search path.  If the search path has not yet been set
 * by a call to <code>setSearchPath</code>, then this fragment will
 * be appended to the path in environment variable SIDL_DLL_PATH.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader_addSearchPath"

void
impl_SIDL_Loader_addSearchPath(
  const char* path_fragment)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader.addSearchPath) */
  if (path_fragment) {
    initialize_search_path();

    if (s_search_path) {
      char* s = SIDL_String_concat3(path_fragment, ";", s_search_path);
      SIDL_String_free(s_search_path);
      s_search_path = s;
    } else {
      s_search_path = SIDL_String_strdup(path_fragment);
    }
  }
  /* DO-NOT-DELETE splicer.end(SIDL.Loader.addSearchPath) */
}

/*
 * Load the specified library if it has not already been loaded.
 * The URI format is defined in class <code>DLL</code>.  The search
 * path is not searched to resolve the library name.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader_loadLibrary"

SIDL_bool
impl_SIDL_Loader_loadLibrary(
  const char* uri)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader.loadLibrary) */
  int ok = FALSE;
  SIDL_DLL dll = SIDL_DLL__create();

  ok = SIDL_DLL_loadLibrary(dll, uri);

  if (ok) {
    impl_SIDL_Loader_addDLL(dll);
  }
  SIDL_DLL_deleteReference(dll);

  return ok;
  /* DO-NOT-DELETE splicer.end(SIDL.Loader.loadLibrary) */
}

/*
 * Append the specified DLL to the beginning of the list of already
 * loaded DLLs.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader_addDLL"

void
impl_SIDL_Loader_addDLL(
  SIDL_DLL dll)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader.addDLL) */
  if (dll) {
    DLL_List* item = NULL;
    initialize_dll_list();
    item = (DLL_List*) malloc(sizeof(DLL_List));
    SIDL_DLL_addReference(dll);
    item->d_dll = dll;
    item->d_next = s_dll_list;
    s_dll_list = item;
  }
  /* DO-NOT-DELETE splicer.end(SIDL.Loader.addDLL) */
}

/*
 * Unload all dynamic link libraries.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader_unloadLibraries"

void
impl_SIDL_Loader_unloadLibraries(
void)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader.unloadLibraries) */
  DLL_List* head = s_dll_list;

  while (head) {
    DLL_List* next = head->d_next;
    SIDL_DLL_deleteReference(head->d_dll);
    free(head);
    head = next;
  }

  s_dll_list = NULL;
  /* DO-NOT-DELETE splicer.end(SIDL.Loader.unloadLibraries) */
}

/*
 * Look up the secified symbol name.  If the symbol name cannot be
 * found in one of the already loaded libraries, then the method will
 * search through the library search path.  A null is returned if the
 * symbol could not be resolved.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader_lookupSymbol"

void*
impl_SIDL_Loader_lookupSymbol(
  const char* linker_name)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader.lookupSymbol) */
  void*     symbol   = NULL;
  DLL_List* dll_item = NULL;

  initialize_dll_list();

  /*
   * Walk current dll list and check whether any DLL defines the symbol
   */
  dll_item = s_dll_list;
  while ((dll_item != NULL) && (symbol == NULL)) {
    symbol = SIDL_DLL_lookupSymbol(dll_item->d_dll, linker_name);
    dll_item = dll_item->d_next;
  }

  /*
   * If the symbol was not found, then search the dll search path
   */
  if (!symbol) {
    char* path  = impl_SIDL_Loader_getSearchPath();
    if (path != NULL) {
      char* entry = strtok(path, ";");
      while ((entry != NULL) && (symbol == NULL)) {

        /*
         * If the URI is not a file, then try to load it directly.
         */
        if (!is_file_uri(entry)) {
          symbol = try_load_dll(entry, linker_name);

        /*
         * If it is a file, then make sure the URI is "file:" with an
         * absolute path using the CWD if necessary.
         */
        } else {
          char* file_uri  = NULL;

          if (SIDL_String_startsWith(entry, "file:/")) {
            file_uri = SIDL_String_strdup(entry);

          } else if (SIDL_String_startsWith(entry, "file:")) {
            char* cwd  = util_getCWD();
            char* file = SIDL_String_substring(entry, 5);
            file_uri   = SIDL_String_concat4("file:", cwd, "/", file);
            SIDL_String_free(cwd);
            SIDL_String_free(file);

          } else if (SIDL_String_startsWith(entry, "/")) {
            file_uri = SIDL_String_concat2("file:", entry);

          } else {
            char* cwd = util_getCWD();
            file_uri  = SIDL_String_concat4("file:", cwd, "/", entry);
            SIDL_String_free(cwd);
          }

          symbol = find_symbol(file_uri, linker_name);
          SIDL_String_free(file_uri);
        }

        if (symbol == NULL) {
          entry = strtok(NULL, ";");
        }
      }
      SIDL_String_free(path);
    }
  }

  return symbol;
  /* DO-NOT-DELETE splicer.end(SIDL.Loader.lookupSymbol) */
}

/*
 * Create an instance of the specified SIDL class.  If the class
 * constructor cannot be found in one of the already loaded libraries,
 * then the method will search through the library search path.  A null
 * object is returned if the symbol could not be resolved.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_Loader_createClass"

SIDL_BaseClass
impl_SIDL_Loader_createClass(
  const char* sidl_name)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader.createClass) */
  SIDL_BaseClass (*ctor)(void) = NULL;

  char* linker = SIDL_String_concat2(sidl_name, "__new");
  char* ptr    = linker;

  while (*ptr) {
    if (*ptr == '.') {
      *ptr = '_';
    }
    ptr++;
  }

  ctor = (SIDL_BaseClass (*)(void)) impl_SIDL_Loader_lookupSymbol(linker);
  SIDL_String_free(linker);

  return ctor ? (*ctor)() : NULL;
  /* DO-NOT-DELETE splicer.end(SIDL.Loader.createClass) */
}
