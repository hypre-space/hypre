/*
 * File:          SIDL_DLL_Impl.c
 * Symbol:        SIDL.DLL-v0.8.1
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for SIDL.DLL
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
 * babel-version = 0.8.0
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "SIDL.DLL" (version 0.8.1)
 * 
 * The <code>DLL</code> class encapsulates access to a single
 * dynamically linked library.  DLLs are loaded at run-time using
 * the <code>loadLibrary</code> method and later unloaded using
 * <code>unloadLibrary</code>.  Symbols in a loaded library are
 * resolved to an opaque pointer by method <code>lookupSymbol</code>.
 * Class instances are created by <code>createClass</code>.
 */

#include "SIDL_DLL_Impl.h"

/* DO-NOT-DELETE splicer.begin(SIDL.DLL._includes) */
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include "SIDL_String.h"
#ifdef HAVE_LIBWWW
#include "wwwconf.h"
#include "WWWLib.h"
#include "WWWInit.h"
#endif

#ifndef NULL
#define NULL 0
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

#ifndef RTLD_LOCAL
#define RTLD_LOCAL 0
#endif

#ifdef HAVE_LIBWWW
static char* url_to_local_file(const char* url)
{
  HTRequest* request = NULL;
#ifdef __CYGWIN__
  char* tmpfile = SIDL_String_concat3("cyg", tmpnam(NULL), ".dll");
#else
  char* tmpfile = SIDL_String_concat3("lib", tmpnam(NULL), ".so");
#endif

  HTProfile_newPreemptiveClient("SIDL_DLL", "1.0");
  HTAlert_setInteractive(FALSE);

  request = HTRequest_new();
  HTLoadToFile(url, request, tmpfile);
  HTRequest_delete(request);
  HTProfile_delete();

  return tmpfile;
}
#endif

/*
 * The libtool dynamic loading library must be initialized before any calls
 * are made to library routines.  Initialize only once.
 */
static void check_lt_initialized(void)
{
  static int initialized = FALSE;
  if (!initialized) {
#ifdef HAVE_LTDL
    (void) lt_dlinit();
#endif
    initialized = TRUE;
  }
}

static int s_sidl_debug_init = 0;
static const char *s_sidl_debug_dlopen = NULL;

static void showLoading(const char *dllname)
{
  fprintf(stderr, "Loading %s: ", dllname);
}

static void showLoadResult(void *handle)
{
  if (handle) {
    fprintf(stderr, "ok\n");
  }
  else {
#ifdef HAVE_LTDL
    const char *errmsg = lt_dlerror(); 
#else
    const char *errmsg = dlerror();
#endif
    fprintf(stderr,"ERROR\n%s\n", errmsg);
  }
}

/* DO-NOT-DELETE splicer.end(SIDL.DLL._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_DLL__ctor"

void
impl_SIDL_DLL__ctor(
  SIDL_DLL self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.DLL._ctor) */
  struct SIDL_DLL__data* data =
    (struct SIDL_DLL__data*) malloc(sizeof (struct SIDL_DLL__data));

  if (!s_sidl_debug_init) {
    s_sidl_debug_dlopen = getenv("SIDL_DEBUG_DLOPEN");
    s_sidl_debug_init = 1;
  }
  data->d_library_handle = NULL;
  data->d_library_name   = NULL;

  SIDL_DLL__set_data(self, data);
  /* DO-NOT-DELETE splicer.end(SIDL.DLL._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_DLL__dtor"

void
impl_SIDL_DLL__dtor(
  SIDL_DLL self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.DLL._dtor) */
  struct SIDL_DLL__data *data = SIDL_DLL__get_data(self);

  impl_SIDL_DLL_unloadLibrary(self);

  free((void*) data);
  SIDL_DLL__set_data(self, NULL);
  /* DO-NOT-DELETE splicer.end(SIDL.DLL._dtor) */
}

/*
 * Load a dynamic link library using the specified URI.  The
 * URI may be of the form "main:", "lib:", "file:", "ftp:", or
 * "http:".  A URI that starts with any other protocol string
 * is assumed to be a file name.  The "main:" URI creates a
 * library that allows access to global symbols in the running
 * program's main address space.  The "lib:X" URI converts the
 * library "X" into a platform-specific name (e.g., libX.so) and
 * loads that library.  The "file:" URI opens the DLL from the
 * specified file path.  The "ftp:" and "http:" URIs copy the
 * specified library from the remote site into a local temporary
 * file and open that file.  This method returns true if the
 * DLL was loaded successfully and false otherwise.  Note that
 * the "ftp:" and "http:" protocols are valid only if the W3C
 * WWW library is available.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_DLL_loadLibrary"

SIDL_bool
impl_SIDL_DLL_loadLibrary(
  SIDL_DLL self, const char* uri)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.DLL.loadLibrary) */
  struct SIDL_DLL__data *data = SIDL_DLL__get_data(self);

  int         ok      = FALSE;
#ifdef HAVE_LTDL
  lt_dlhandle handle  = NULL;
#else
  void*       handle  = NULL;
#endif
  char*       dllfile = NULL;
  char*       dllname = NULL;

  if (data->d_library_handle) {
    impl_SIDL_DLL_unloadLibrary(self);
  }

  if (SIDL_String_equals(uri, "main:")) {
    dllfile = NULL;
    dllname = SIDL_String_strdup(uri);

  } else if (SIDL_String_startsWith(uri, "lib:")) {
    char* dll = SIDL_String_substring(uri, 4);
#ifdef HAVE_LTDL
    dllfile = SIDL_String_concat3("lib", dll, ".la");
#else
#ifdef __CYGWIN__
    dllfile = SIDL_String_concat3("cyg", dll, ".dll");
#else
    dllfile = SIDL_String_concat3("lib", dll, ".so");
#endif
#endif
    dllname = SIDL_String_strdup(uri);
    SIDL_String_free(dll);

  } else if (SIDL_String_startsWith(uri, "file:")) {
    dllfile = SIDL_String_substring(uri, 5);
    dllname = SIDL_String_strdup(uri);

#ifdef HAVE_LIBWWW
  } else if (SIDL_String_startsWith(uri, "ftp:")) {
    dllfile = url_to_local_file(uri);
    dllname = SIDL_String_strdup(uri);

  } else if (SIDL_String_startsWith(uri, "http:")) {
    dllfile = url_to_local_file(uri);
    dllname = SIDL_String_strdup(uri);
#endif

  } else {
    dllfile = SIDL_String_strdup(uri);
    dllname = SIDL_String_concat2("file:", uri);
  }

  if (s_sidl_debug_dlopen) showLoading(dllfile);
#ifdef HAVE_LTDL
  check_lt_initialized();
  handle = lt_dlopen(dllfile);
#else
  handle = dlopen(dllfile, RTLD_LAZY | RTLD_LOCAL);
#endif
  if (s_sidl_debug_dlopen) showLoadResult((void *)handle);
  SIDL_String_free(dllfile);

  if (handle) {
    ok = TRUE;
    data->d_library_handle = handle;
    data->d_library_name   = dllname;
  } else {
    ok = FALSE;
    SIDL_String_free(dllname);
  }

  return ok;
  /* DO-NOT-DELETE splicer.end(SIDL.DLL.loadLibrary) */
}

/*
 * Get the library name.  This is the name used to load the
 * library in <code>loadLibrary</code> except that all file names
 * contain the "file:" protocol.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_DLL_getName"

char*
impl_SIDL_DLL_getName(
  SIDL_DLL self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.DLL.getName) */
  struct SIDL_DLL__data *data = SIDL_DLL__get_data(self);

  char* name = NULL;
  if (data->d_library_name) {
    name = SIDL_String_strdup(data->d_library_name);
  }
  return name;
  /* DO-NOT-DELETE splicer.end(SIDL.DLL.getName) */
}

/*
 * Unload the dynamic link library.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_DLL_unloadLibrary"

void
impl_SIDL_DLL_unloadLibrary(
  SIDL_DLL self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.DLL.unloadLibrary) */
  struct SIDL_DLL__data *data = SIDL_DLL__get_data(self);

  if (data->d_library_handle) {
#ifdef HAVE_LTDL
    (void) lt_dlclose(data->d_library_handle);
#else
    (void) dlclose(data->d_library_handle);
#endif
    SIDL_String_free(data->d_library_name);
    data->d_library_handle = NULL;
    data->d_library_name   = NULL;
  }
  /* DO-NOT-DELETE splicer.end(SIDL.DLL.unloadLibrary) */
}

/*
 * Lookup a symbol from the DLL and return the associated pointer.
 * A null value is returned if the name does not exist.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_DLL_lookupSymbol"

void*
impl_SIDL_DLL_lookupSymbol(
  SIDL_DLL self, const char* linker_name)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.DLL.lookupSymbol) */
  struct SIDL_DLL__data *data = SIDL_DLL__get_data(self);

  void* address = NULL;
  if (data->d_library_handle) {
#ifdef HAVE_LTDL
    address = (void*) lt_dlsym(data->d_library_handle, linker_name);
#else
    address = dlsym(data->d_library_handle, linker_name);
#endif
  }
  return address;
  /* DO-NOT-DELETE splicer.end(SIDL.DLL.lookupSymbol) */
}

/*
 * Create an instance of the SIDL class.  If the class constructor
 * is not defined in this DLL, then return null.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_DLL_createClass"

SIDL_BaseClass
impl_SIDL_DLL_createClass(
  SIDL_DLL self, const char* sidl_name)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.DLL.createClass) */
  struct SIDL_DLL__data *data = SIDL_DLL__get_data(self);
  SIDL_BaseClass (*ctor)(void) = NULL;

  if (data->d_library_handle) {
    char* linker = SIDL_String_concat2(sidl_name, "__new");
    char* ptr    = linker;

    while (*ptr) {
      if (*ptr == '.') {
        *ptr = '_';
      }
      ptr++;
    }

#ifdef HAVE_LTDL
    ctor = (SIDL_BaseClass (*)(void)) lt_dlsym(data->d_library_handle, linker);
#else
    ctor = (SIDL_BaseClass (*)(void)) dlsym(data->d_library_handle, linker);
#endif
    SIDL_String_free(linker);
  }

  return ctor ? (*ctor)() : NULL;
  /* DO-NOT-DELETE splicer.end(SIDL.DLL.createClass) */
}
