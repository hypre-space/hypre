/*
 * File:          sidlx_rmi_SimpleTicket_Impl.h
 * Symbol:        sidlx.rmi.SimpleTicket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for sidlx.rmi.SimpleTicket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_sidlx_rmi_SimpleTicket_Impl_h
#define included_sidlx_rmi_SimpleTicket_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_Ticket_h
#include "sidl_rmi_Ticket.h"
#endif
#ifndef included_sidl_rmi_TicketBook_h
#include "sidl_rmi_TicketBook.h"
#endif
#ifndef included_sidlx_rmi_SimpleTicket_h
#include "sidlx_rmi_SimpleTicket.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicket._includes) */
#include "sidlx_rmi_Simsponse.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicket._includes) */

/*
 * Private data for class sidlx.rmi.SimpleTicket
 */

struct sidlx_rmi_SimpleTicket__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicket._data) */
  sidlx_rmi_Simsponse d_response;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicket._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_SimpleTicket__data*
sidlx_rmi_SimpleTicket__get_data(
  sidlx_rmi_SimpleTicket);

extern void
sidlx_rmi_SimpleTicket__set_data(
  sidlx_rmi_SimpleTicket,
  struct sidlx_rmi_SimpleTicket__data*);

extern
void
impl_sidlx_rmi_SimpleTicket__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleTicket__ctor(
  /* in */ sidlx_rmi_SimpleTicket self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleTicket__ctor2(
  /* in */ sidlx_rmi_SimpleTicket self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleTicket__dtor(
  /* in */ sidlx_rmi_SimpleTicket self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_rmi_Response(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_rmi_Response(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_rmi_Ticket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_rmi_Ticket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_TicketBook__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_rmi_TicketBook(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_TicketBook__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_rmi_TicketBook(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleTicket__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidlx_rmi_SimpleTicket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleTicket__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidlx_rmi_SimpleTicket(void* bi, 
  sidl_BaseInterface* _ex);
extern
void
impl_sidlx_rmi_SimpleTicket_setResponse(
  /* in */ sidlx_rmi_SimpleTicket self,
  /* in */ sidl_rmi_Response r,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleTicket_block(
  /* in */ sidlx_rmi_SimpleTicket self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_SimpleTicket_test(
  /* in */ sidlx_rmi_SimpleTicket self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_TicketBook
impl_sidlx_rmi_SimpleTicket_createEmptyTicketBook(
  /* in */ sidlx_rmi_SimpleTicket self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_Response
impl_sidlx_rmi_SimpleTicket_getResponse(
  /* in */ sidlx_rmi_SimpleTicket self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_rmi_Response(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_rmi_Response(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_rmi_Ticket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_rmi_Ticket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_TicketBook__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidl_rmi_TicketBook(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_TicketBook__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidl_rmi_TicketBook(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleTicket__object* 
  impl_sidlx_rmi_SimpleTicket_fconnect_sidlx_rmi_SimpleTicket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleTicket__object* 
  impl_sidlx_rmi_SimpleTicket_fcast_sidlx_rmi_SimpleTicket(void* bi, 
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
