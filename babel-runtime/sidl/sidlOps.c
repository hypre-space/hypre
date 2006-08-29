#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sidlOps.h"
#include "sidl_BaseInterface.h"

#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"


/*
 * Dynamically load the named symbol.
 */

void* sidl_dynamicLoadIOR(char* objName, char* extName) {
  sidl_BaseInterface _throwaway_exception = NULL; /*(TODO: a way to not throw these away? */

  void * _locExternals = NULL;
  sidl_DLL dll = sidl_DLL__create(&_throwaway_exception);
  void* (*dll_f)(void);
  /* check global namespace for symbol first */
  if (dll && sidl_DLL_loadLibrary(dll, "main:", TRUE, FALSE,&_throwaway_exception)) {
    dll_f =
      (  void*(*)(void)) sidl_DLL_lookupSymbol(
					       dll, extName,&_throwaway_exception);
    _locExternals = (dll_f ? (*dll_f)() : NULL);
  }
  if (dll) sidl_DLL_deleteRef(dll,&_throwaway_exception);
  if (!_locExternals) {
    dll = sidl_Loader_findLibrary(objName,
				  "ior/impl", sidl_Scope_SCLSCOPE,
				  sidl_Resolve_SCLRESOLVE,&_throwaway_exception);
    if (dll) {
      dll_f =
        (  void*(*)(void)) sidl_DLL_lookupSymbol(
						 dll, extName,&_throwaway_exception);
      _locExternals = (dll_f ? (*dll_f)() : NULL);
      sidl_DLL_deleteRef(dll,&_throwaway_exception);
    }
  }
  if (!_locExternals) {
    fputs("Babel: unable to load the implementation for ", stderr);
    fputs(objName, stderr);
    fputs(" please set SIDL_DLL_PATH\n", stderr);
    exit(-1);
  }
  return _locExternals;
}
#endif

void
sidl_checkIORVersion(const char *libraryOrType,
                     int libMajor, int libMinor,
                     int progMajor, int progMinor)
{
  if (libMajor != progMajor) {
    fprintf(stderr, "babel: ERROR IOR version mismatch (library IOR version %d.%d, program IOR version %d.%d) for library/type %s\n",
            libMajor, libMinor, progMajor, progMinor, libraryOrType);
    exit(2); /* this is serious enough that the program should exit */
  }
  if (libMinor != progMinor) {
    fprintf(stderr, "babel: WARNING minor IOR version mismatch (library IOR version %d.%d, program IOR version %d.%d) for library/type %s\n",
            libMajor, libMinor, progMajor, progMinor, libraryOrType);
    /* do not exit */
  }
}

struct sidl_atexit_list {
  sidl_atexit_func         d_func;
  void                    *d_data;
  struct sidl_atexit_list *d_next;
};

struct sidl_atexit_list *g_atexit_list = NULL;

static void 
sidl_atexit_impl(void)
{
  struct sidl_atexit_list *tmp;
  while (g_atexit_list != NULL) {
    tmp = g_atexit_list->d_next;
    (*(g_atexit_list->d_func))(g_atexit_list->d_data);
    free((void *)g_atexit_list);
    g_atexit_list = tmp;
  }
}

void sidl_atexit(sidl_atexit_func fcn, void *data)
{
  struct sidl_atexit_list * const tmp = g_atexit_list;
  if (!tmp) {
    atexit(sidl_atexit_impl);
  }
  g_atexit_list = malloc(sizeof(struct sidl_atexit_list));
  g_atexit_list->d_next = tmp;
  g_atexit_list->d_data = data;
  g_atexit_list->d_func = fcn;
}

void
sidl_deleteRef_atexit(void *objref)
{
  sidl_BaseInterface *obj=
    (sidl_BaseInterface *)objref;
  sidl_BaseInterface ignored;
  if (obj && *obj) {
    sidl_BaseInterface_deleteRef(*obj, &ignored);
    *obj = NULL;
  }
}
