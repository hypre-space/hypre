/*
 * File:        sidlOps.h
 * Copyright:   (c) 2005 The Regents of the University of California
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Special options that are common through out babel.
 *
 */

#ifndef included_sidlOps_h
#define included_sidlOps_h
#include "babel_config.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef SIDL_DYNAMIC_LIBRARY
  void* sidl_dynamicLoadIOR(char* objName, char* extName);
#endif

/**
 * Check if libMajor and libMinor are consistent with progMajor
 * and progMinor. A mismatch on major will cause the program
 * to exit. A mismatch on minor will cause a warning message.
 * libraryOrType should indicate the name of the shared library
 * or type where the mismatch may have taken place.
 */
void
sidl_checkIORVersion(const char *libraryOrType,
                     int libMajor, int libMinor,
                     int progMajor, int progMinor);

/**
 * Register a function to be called during normal program termination
 * via the ANSI Standard C atexit() function. The documentation for
 * atexit implies that the number of functions that can be registered
 * may be limited. This routine enables Babel to have several functions
 * called during normal program terminal while only taking one slot
 * in the standard atexit() list.
 *
 * Functions are called in the reverse order of their registration.
 * The data pointer passed to babel_atexit is passed to fcn when it's
 * called.
 */
typedef void (*sidl_atexit_func)(void *);
void sidl_atexit(sidl_atexit_func fcn, void *data);

/**
 * This is a convenience function to call deleteRef on a sidl.BaseInterface
 * during normal program termination. It's intended to be used with
 * sidl_atexit.
 *
 * The true signature for this function is:
 * void
 * sidl_deleteRef_atexit(struct sidl_BaseInterface__object **objref)
 * {
 *   if (objref && *objref) {
 *      sidl_BaseInterface_deleteRef(*objref);
 *      *objref = NULL;
 *   }
 * }
 */
void sidl_deleteRef_atexit(void *objref);

#ifdef __cplusplus
}
#endif
#endif /*  included_sidlOps_h */
