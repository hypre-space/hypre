/*
 * File:        sidl_Python.c
 * Release:     $Name$
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


void sidl_Python_Init(void)
{
  static int python_notinitialized = 1;
#ifdef PYTHON_SHARED_LIBRARY
  static const char libName[] = PYTHON_SHARED_LIBRARY;
#endif
  sidl_DLL dll;
  static const char initName[] = "Py_Initialize";
  static const char finalName[] = "Py_Finalize";
  void (*pyinit)(void);
  void *handle;
  if (python_notinitialized) {
    dll = sidl_Loader_loadLibrary("main:", TRUE, TRUE);
    if (dll) {
      pyinit = (void (*)(void))sidl_DLL_lookupSymbol(dll, initName);
      if (pyinit) {
        (*pyinit)();
        python_notinitialized = 0;
        pyinit = (void (*)(void))sidl_DLL_lookupSymbol(dll, finalName);
        if (pyinit) {
          atexit(pyinit);
        }
      }
      sidl_DLL_deleteRef(dll);
    }

    if (python_notinitialized) {
#ifdef PYTHON_SHARED_LIBRARY
      char *url = sidl_String_concat2("file:", PYTHON_SHARED_LIBRARY);
      if (url) {
        dll = sidl_Loader_loadLibrary(url, TRUE, TRUE);
        if (dll) {
          pyinit = (void (*)(void))sidl_DLL_lookupSymbol(dll, initName);
          if (pyinit) {
            python_notinitialized = 0;
            (*pyinit)();
            pyinit = (void (*)(void))sidl_DLL_lookupSymbol(dll, finalName);
            if (pyinit) {
              atexit(pyinit);
            }
          } 
          else {
            fprintf(stderr, "Babel: Error: Unable to find symbol %s in %s",
                    initName, libName);
          }
          sidl_DLL_deleteRef(dll);
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
