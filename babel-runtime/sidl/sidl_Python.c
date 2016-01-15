/*
 * File:        sidl_Python.c
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Initialize a Python language interpretter
 *
 */

#include "sidl_Python.h"
#include "babel_config.h"
#ifndef included_sidl_DLL_h
#include "sidl_DLL.h"
#endif
#ifndef included_sidl_Loader_h
#include "sidl_Loader.h"
#endif
#ifndef included_sidl_String_h
#include "sidl_String.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include "sidlOps.h"

static void (*g_python_shutdown)(void) = NULL;

static void
sidl_Python_shutdown(void *ignored)
{
  if (g_python_shutdown) {
    (*g_python_shutdown)();
  }
}

void sidl_Python_Init(void)
{
  sidl_BaseInterface throwaway_exception; /*TODO: a way to not throw these away? */
  static int python_notinitialized = 1;
#ifdef PYTHON_SHARED_LIBRARY
  static const char libName[] = PYTHON_SHARED_LIBRARY;
#endif
  sidl_DLL dll;
  static const char initName[] = "Py_Initialize";
  static const char finalName[] = "Py_Finalize";
  void (*pyinit)(void);
  if (python_notinitialized) {
    dll = sidl_Loader_loadLibrary("main:", TRUE, TRUE, &throwaway_exception);
    if (dll) {
      pyinit = (void (*)(void))sidl_DLL_lookupSymbol(dll, initName,&throwaway_exception);
      if (pyinit) {
        (*pyinit)();
        python_notinitialized = 0;
        g_python_shutdown = (void (*)(void))sidl_DLL_lookupSymbol(dll, finalName,&throwaway_exception);
        if (g_python_shutdown) {
          sidl_atexit(sidl_Python_shutdown, NULL);
        }
      }
      sidl_DLL_deleteRef(dll,&throwaway_exception);
    }

    if (python_notinitialized) {
#ifdef PYTHON_SHARED_LIBRARY
      char *url = sidl_String_concat2("file:", PYTHON_SHARED_LIBRARY);
      if (url) {
        dll = sidl_Loader_loadLibrary(url, TRUE, TRUE,&throwaway_exception);
        if (dll) {
          pyinit = (void (*)(void))sidl_DLL_lookupSymbol(dll, initName,&throwaway_exception);
          if (pyinit) {
            python_notinitialized = 0;
            (*pyinit)();
            g_python_shutdown = (void (*)(void))sidl_DLL_lookupSymbol(dll, finalName,&throwaway_exception);
            if (g_python_shutdown) {
              sidl_atexit(sidl_Python_shutdown, NULL);
            }
          } 
          else {
            fprintf(stderr, "Babel: Error: Unable to find symbol %s in %s",
                    initName, libName);
          }
          sidl_DLL_deleteRef(dll,&throwaway_exception);
        }
        else {
          fprintf(stderr,
                  "Babel: Error: Unable to load library %s\n", libName);
        }
        sidl_String_free(url);
      }
      else {
        fprintf(stderr, "Unable to allocate string or sidl.DDL object\n");
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
