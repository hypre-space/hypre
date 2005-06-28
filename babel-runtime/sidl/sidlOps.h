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

#ifdef __cplusplus
}
#endif
#endif /*  included_sidlOps_h */
