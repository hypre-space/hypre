/*
 * File:          sidlx_rmi_CallType_IOR.h
 * Symbol:        sidlx.rmi.CallType-v0.1
 * Symbol Type:   enumeration
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for sidlx.rmi.CallType
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
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
 * There are 3 basic types of calls on the server side, CREATE, EXEC, and CONNECT.
 * This definese them 
 */


/* Opaque forward declaration of array struct */
struct sidlx_rmi_CallType__array;

enum sidlx_rmi_CallType__enum {
  /**
   * Create and register a new instance. 
   */
  sidlx_rmi_CallType_CREATE  = 0,

  /**
   * Call a method. 
   */
  sidlx_rmi_CallType_EXEC    = 1,

  /**
   * Connect to an existing instance. 
   */
  sidlx_rmi_CallType_CONNECT = 2

};

#ifdef __cplusplus
}
#endif
#endif
