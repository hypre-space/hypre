/*
 * File:          sidlx_rmi_SimpleTicket_IOR.h
 * Symbol:        sidlx.rmi.SimpleTicket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Intermediate Object Representation for sidlx.rmi.SimpleTicket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_sidlx_rmi_SimpleTicket_IOR_h
#define included_sidlx_rmi_SimpleTicket_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif
#ifndef included_sidl_rmi_Ticket_IOR_h
#include "sidl_rmi_Ticket_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.SimpleTicket" (version 0.1)
 * 
 * implementation of a Ticket 
 */

struct sidlx_rmi_SimpleTicket__array;
struct sidlx_rmi_SimpleTicket__object;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_RuntimeException__array;
struct sidl_RuntimeException__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_Response__array;
struct sidl_rmi_Response__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;
struct sidl_rmi_TicketBook__array;
struct sidl_rmi_TicketBook__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_rmi_SimpleTicket__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in sidl.rmi.Ticket-v0.9.15 */
  void (*f_block)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_test)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_rmi_TicketBook__object* (*f_createEmptyTicketBook)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_rmi_Response__object* (*f_getResponse)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi.SimpleTicket-v0.1 */
  void (*f_setResponse)(
    /* in */ struct sidlx_rmi_SimpleTicket__object* self,
    /* in */ struct sidl_rmi_Response__object* r,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the controls structure.
 */


struct sidlx_rmi_SimpleTicket__controls {
  int     use_hooks;
};
/*
 * Define the class object structure.
 */

struct sidlx_rmi_SimpleTicket__object {
  struct sidl_BaseClass__object       d_sidl_baseclass;
  struct sidl_rmi_Ticket__object      d_sidl_rmi_ticket;
  struct sidlx_rmi_SimpleTicket__epv* d_epv;
  void*                               d_data;
};

struct sidlx_rmi_SimpleTicket__external {
  struct sidlx_rmi_SimpleTicket__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_SimpleTicket__external*
sidlx_rmi_SimpleTicket__externals(void);

extern struct sidlx_rmi_SimpleTicket__object*
sidlx_rmi_SimpleTicket__new(void* ddata,struct sidl_BaseInterface__object ** 
  _ex);

extern void sidlx_rmi_SimpleTicket__init(
  struct sidlx_rmi_SimpleTicket__object* self, void* ddata, struct 
    sidl_BaseInterface__object ** _ex);
extern void sidlx_rmi_SimpleTicket__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,struct 
    sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidl_rmi_Ticket__epv **s_arg_epv__sidl_rmi_ticket,
  struct sidl_rmi_Ticket__epv **s_arg_epv_hooks__sidl_rmi_ticket,
  struct sidlx_rmi_SimpleTicket__epv **s_arg_epv__sidlx_rmi_simpleticket,struct 
    sidlx_rmi_SimpleTicket__epv **s_arg_epv_hooks__sidlx_rmi_simpleticket);
  extern void sidlx_rmi_SimpleTicket__fini(
    struct sidlx_rmi_SimpleTicket__object* self, struct 
      sidl_BaseInterface__object ** _ex);
  extern void sidlx_rmi_SimpleTicket__IOR_version(int32_t *major, int32_t 
    *minor);

  struct sidl_BaseClass__object* 
    skel_sidlx_rmi_SimpleTicket_fconnect_sidl_BaseClass(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* 
    skel_sidlx_rmi_SimpleTicket_fcast_sidl_BaseClass(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_sidlx_rmi_SimpleTicket_fconnect_sidl_BaseInterface(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_sidlx_rmi_SimpleTicket_fcast_sidl_BaseInterface(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_sidlx_rmi_SimpleTicket_fconnect_sidl_ClassInfo(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* 
    skel_sidlx_rmi_SimpleTicket_fcast_sidl_ClassInfo(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_sidlx_rmi_SimpleTicket_fconnect_sidl_RuntimeException(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_sidlx_rmi_SimpleTicket_fcast_sidl_RuntimeException(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_rmi_Response__object* 
    skel_sidlx_rmi_SimpleTicket_fconnect_sidl_rmi_Response(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_Response__object* 
    skel_sidlx_rmi_SimpleTicket_fcast_sidl_rmi_Response(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_rmi_Ticket__object* 
    skel_sidlx_rmi_SimpleTicket_fconnect_sidl_rmi_Ticket(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_Ticket__object* 
    skel_sidlx_rmi_SimpleTicket_fcast_sidl_rmi_Ticket(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_rmi_TicketBook__object* 
    skel_sidlx_rmi_SimpleTicket_fconnect_sidl_rmi_TicketBook(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_TicketBook__object* 
    skel_sidlx_rmi_SimpleTicket_fcast_sidl_rmi_TicketBook(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_SimpleTicket__object* 
    skel_sidlx_rmi_SimpleTicket_fconnect_sidlx_rmi_SimpleTicket(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidlx_rmi_SimpleTicket__object* 
    skel_sidlx_rmi_SimpleTicket_fcast_sidlx_rmi_SimpleTicket(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_SimpleTicket__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

#ifdef __cplusplus
  }
#endif
#endif
