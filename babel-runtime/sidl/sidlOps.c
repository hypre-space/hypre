#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sidlOps.h"

#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"


/*
 * Dynamically load the named symbol.
 */

void* sidl_dynamicLoadIOR(char* objName, char* extName) {
  void * _locExternals = NULL;
  sidl_DLL dll = sidl_DLL__create();
  void* (*dll_f)(void);
  /* check global namespace for symbol first */
  if (dll && sidl_DLL_loadLibrary(dll, "main:", TRUE, FALSE)) {
    dll_f =
      (  void*(*)(void)) sidl_DLL_lookupSymbol(
					       dll, extName);
    _locExternals = (dll_f ? (*dll_f)() : NULL);
  }
  if (dll) sidl_DLL_deleteRef(dll);
  if (!_locExternals) {
    dll = sidl_Loader_findLibrary(objName,
				  "ior/impl", sidl_Scope_SCLSCOPE,
				  sidl_Resolve_SCLRESOLVE);
    if (dll) {
      dll_f =
        (  void*(*)(void)) sidl_DLL_lookupSymbol(
						 dll, extName);
      _locExternals = (dll_f ? (*dll_f)() : NULL);
      sidl_DLL_deleteRef(dll);
    }
  }
  if (!_locExternals) {
    fputs("Babel: unable to load the implementation for ", stderr);
    fputs(objName, stderr);
    fputs(" please set sidl_DLL_PATH\n", stderr);
    exit(-1);
  }
  return _locExternals;
}
#endif

