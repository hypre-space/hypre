/*
 * File:        SIDL_Python.c
 * Release:     $Name: V1-9-0b $
 * Revision:    @(#) $Revision: 1.4 $
 * Date:        $Date: 2003/04/07 21:44:31 $
 * Description: Initialize a Python language interpretter
 *

This includes excerpts from: 
   ltdl.c -- system independent dlopen wrapper
   Copyright (C) 1998, 1999, 2000 Free Software Foundation, Inc.
   Originally by Thomas Tanner <tanner@ffii.org>
   This file is part of GNU Libtool.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

As a special exception to the GNU Lesser General Public License,
if you distribute this file as part of a program or library that
is built using GNU libtool, you may include it under the same
distribution terms that you use for the rest of that program.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
02111-1307  USA
*/

#include "SIDL_Python.h"
#include "babel_config.h"
#include <stdio.h>
#include <stdlib.h>

#if HAVE_LIBDL

/* dynamic linking with dlopen/dlsym */

#if HAVE_DLFCN_H
#  include <dlfcn.h>
#endif

#if HAVE_SYS_DL_H
#  include <sys/dl.h>
#endif

#ifdef RTLD_GLOBAL
#  define LT_GLOBAL		RTLD_GLOBAL
#else
#  ifdef DL_GLOBAL
#    define LT_GLOBAL		DL_GLOBAL
#  endif
#endif /* !RTLD_GLOBAL */
#ifndef LT_GLOBAL
#  define LT_GLOBAL		0
#endif /* !LT_GLOBAL */

/* We may have to define LT_LAZY_OR_NOW in the command line if we
   find out it does not work in some platform. */
#ifndef LT_LAZY_OR_NOW
#  ifdef RTLD_LAZY
#    define LT_LAZY_OR_NOW	RTLD_LAZY
#  else
#    ifdef DL_LAZY
#      define LT_LAZY_OR_NOW	DL_LAZY
#    endif
#  endif /* !RTLD_LAZY */
#endif
#ifndef LT_LAZY_OR_NOW
#  ifdef RTLD_NOW
#    define LT_LAZY_OR_NOW	RTLD_NOW
#  else
#    ifdef DL_NOW
#      define LT_LAZY_OR_NOW	DL_NOW
#    endif
#  endif /* !RTLD_NOW */
#endif
#ifndef LT_LAZY_OR_NOW
#  define LT_LAZY_OR_NOW	0
#endif /* !LT_LAZY_OR_NOW */

void SIDL_Python_Init(void)
{
  static int python_notinitialized = 1;
#ifdef PYTHON_SHARED_LIBRARY
  static const char libName[] = PYTHON_SHARED_LIBRARY;
#endif
  static const char initName[] = "Py_Initialize";
  static const char finalName[] = "Py_Finalize";
  void (*pyinit)(void);
  void *handle;
  if (python_notinitialized) {
    /* search the previous loaded global namespace */
    handle = dlopen(0, LT_GLOBAL | LT_LAZY_OR_NOW);
    if (handle) {
      pyinit = (void (*)(void))dlsym(handle, initName);
      if (pyinit) {
        (*pyinit)();
        python_notinitialized = 0;
        pyinit = (void (*)(void))dlsym(handle, finalName);
        if (pyinit) {
          atexit(pyinit);
        }
      }
    }

    if (python_notinitialized) {
#ifdef PYTHON_SHARED_LIBRARY
      handle = dlopen(libName, LT_GLOBAL | LT_LAZY_OR_NOW);
      if (handle) {
        pyinit = (void (*)(void))dlsym(handle, initName);
        if (pyinit) {
          python_notinitialized = 0;
          (*pyinit)();
          pyinit = (void (*)(void))dlsym(handle, finalName);
          if (pyinit) {
            atexit(pyinit);
          }
          return;
        } else {
          fprintf(stderr, "Babel: Error: Unable to find symbol %s in %s: %s",
                  initName, libName, dlerror());
          dlclose(handle);
        }
      } else {
        fprintf(stderr, "Babel: Error: Unable to load library %s: %s", libName, dlerror());
      }
#else
      fprintf(stderr, "Babel: Error: Unable to initialize Python.\n\
The BABEL runtime library was not configured for Python support,\n\
and Python is not already loaded into the global symbol space.\n");
      python_notinitialized = 0;
#endif
    }
  }
}

#else

void SIDL_Python_Init(void)
{
  static int python_notinitialized = 1;
  fprintf(stderr, "Babel: Error: Unable to initialize Python.\n\
The BABEL runtime is not able to initialize Python because it\n\
has no support for loading shared libraries.\n");
  python_notinitialized = 0;
  
}
#endif
