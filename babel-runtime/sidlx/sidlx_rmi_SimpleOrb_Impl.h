/*
 * File:          sidlx_rmi_SimpleOrb_Impl.h
 * Symbol:        sidlx.rmi.SimpleOrb-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.rmi.SimpleOrb
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.6
 */

#ifndef included_sidlx_rmi_SimpleOrb_Impl_h
#define included_sidlx_rmi_SimpleOrb_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidlx_rmi_SimpleOrb_h
#include "sidlx_rmi_SimpleOrb.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._includes) */

/*
 * Private data for class sidlx.rmi.SimpleOrb
 */

struct sidlx_rmi_SimpleOrb__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_SimpleOrb__data*
sidlx_rmi_SimpleOrb__get_data(
  sidlx_rmi_SimpleOrb);

extern void
sidlx_rmi_SimpleOrb__set_data(
  sidlx_rmi_SimpleOrb,
  struct sidlx_rmi_SimpleOrb__data*);

extern void
impl_sidlx_rmi_SimpleOrb__ctor(
  sidlx_rmi_SimpleOrb);

extern void
impl_sidlx_rmi_SimpleOrb__dtor(
  sidlx_rmi_SimpleOrb);

/*
 * User-defined object methods
 */

extern void
impl_sidlx_rmi_SimpleOrb_serviceRequest(
  sidlx_rmi_SimpleOrb,
  int32_t,
  sidl_BaseInterface*);

#ifdef __cplusplus
}
#endif
#endif
