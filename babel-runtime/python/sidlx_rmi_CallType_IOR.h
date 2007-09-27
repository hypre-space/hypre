/*
 * File:          sidlx_rmi_CallType_IOR.h
 * Symbol:        sidlx.rmi.CallType-v0.1
 * Symbol Type:   enumeration
 * Babel Version: 1.0.4
 * Description:   Intermediate Object Representation for sidlx.rmi.CallType
 * 
 * WARNING: Automatically generated; changes will be lost
 */

#ifndef included_sidlx_rmi_CallType_IOR_h
#define included_sidlx_rmi_CallType_IOR_h

#ifndef included_sidlType_h
#include "sidlType.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.CallType" (version 0.1)
 * 
 * There are 3 basic types of calls on the server side, CREATE, EXEC, and SERIAL.
 * This enumeration defines them.  
 */


/* Opaque forward declaration of array struct */
struct sidlx_rmi_CallType__array;

enum sidlx_rmi_CallType__enum {
  sidlx_rmi_CallType_CREATE = 0,

  sidlx_rmi_CallType_EXEC   = 1,

  sidlx_rmi_CallType_SERIAL = 2

};

#ifdef __cplusplus
}
#endif
#endif
