/*
 * File:          sidlx_rmi_SimCall_Impl.c
 * Symbol:        sidlx.rmi.SimCall-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for sidlx.rmi.SimCall
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.SimCall" (version 0.1)
 * 
 * This type is created on the server side to get inargs off the network and 
 * pass them into exec.	
 */

#include "sidlx_rmi_SimCall_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall._includes) */
#include "sidlx_rmi_GenNetworkException.h"
#include "sidlType.h"
#include "sidl_Exception.h"
#include "sidl_String.h"
/** Parses string into tokens, replaces token seperator with '\0' and
 *  returns the pointer to the beginning of this token.  Should only be used
 *  when you know you're dealing with an alpha-numeric string.
 */

static char* get_next_token(sidlx_rmi_SimCall self,/*out*/ sidl_BaseInterface* _ex) {
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr){
    /* int counter = dptr->d_current; */
    int upper = sidl_char__array_upper(dptr->d_carray,0);
    char* d_buf = sidl_char__array_first(dptr->d_carray);
    char* begin = d_buf+dptr->d_current;
    char* s_ptr = begin;

    while(*s_ptr != ':') {
      ++s_ptr;
      ++(dptr->d_current);
      if(*s_ptr == '\0' || dptr->d_current > upper) {
	SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.get_next_token:Improperly formed response!");  
      }
    }
    *s_ptr = '\0';
    ++(dptr->d_current); /* Advance the the beginning of the next token */
    return begin;
  EXIT:
    return NULL;
  }
}

static void unserialize(sidlx_rmi_SimCall self, char* data, int n, sidl_BaseInterface* _ex) {
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  int i = 0;
  char* d_buf = sidl_char__array_first(dptr->d_carray);
  int d_capacity = sidl_char__array_length(dptr->d_carray, 0);
  int rem = d_capacity - dptr->d_current; /*space remaining*/
  char* s_ptr = (d_buf)+(dptr->d_current);
  if(n>rem) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.unserialize: Not enough data left!");  
  }
  memcpy(data, s_ptr, n);
  (dptr->d_current) += n;
 EXIT:
  return;
}

static void flip64(int64_t* in) {
  int64_t x = *in;
  *in =  ((((x) & 0xff00000000000000ull) >> 56)				
	  | (((x) & 0x00ff000000000000ull) >> 40)			
	  | (((x) & 0x0000ff0000000000ull) >> 24)			
	  | (((x) & 0x000000ff00000000ull) >> 8)			
	  | (((x) & 0x00000000ff000000ull) << 8)			
	  | (((x) & 0x0000000000ff0000ull) << 24)			
	  | (((x) & 0x000000000000ff00ull) << 40)			
	  | (((x) & 0x00000000000000ffull) << 56));
}

static void flip32(int32_t* in) {
  int32_t x = *in;
  *in = ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |	
	 (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24));
}


/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall._load) */
  /* insert implementation here: sidlx.rmi.SimCall._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall__ctor(
  /* in */ sidlx_rmi_SimCall self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall._ctor) */
  /* insert implementation here: sidlx.rmi.SimCall._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall__dtor(
  /* in */ sidlx_rmi_SimCall self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall._dtor) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr) {
    sidl_char__array_deleteRef(dptr->d_carray);
    sidlx_rmi_Socket_deleteRef(dptr->d_sock);
    sidl_String_free(dptr->d_methodName);
    sidl_String_free(dptr->d_clsid);
    sidl_String_free(dptr->d_objid);
    free((void*)dptr);
    /* FIXME:    struct sidlx_rmi_SimReturn__data *dptr =
       sidlx_rmi_SimReturn__set_data(self, NULL); */
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall._dtor) */
}

/*
 * Method:  init[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_init"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_init(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.init) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  char* token = NULL;
  if (dptr) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "This Call has already been init'ed!");
  } else {
    dptr = malloc(sizeof(struct sidlx_rmi_SimCall__data));
  }
  dptr->d_methodName = NULL;
  dptr->d_clsid = NULL;
  dptr->d_objid = NULL;
  dptr->d_sock = sock;
  dptr->d_carray = NULL;
  dptr->d_current = 0;
  sidlx_rmi_SimCall__set_data(self, dptr);

  sidlx_rmi_Socket_readstring_alloc(sock,&(dptr->d_carray),_ex);SIDL_CHECK(*_ex);

  token = get_next_token(self, _ex); SIDL_CHECK(*_ex);
  if(sidl_String_equals(token, "CREATE")) {
    char * type = NULL;
    dptr->d_calltype = sidlx_rmi_CallType_CREATE;
    dptr->d_objid = NULL;
    dptr->d_methodName = sidl_String_strdup("CREATE");
    sidlx_rmi_SimCall_unpackString(self, "className", &(dptr->d_clsid), _ex); SIDL_CHECK(*_ex);
  } else if(sidl_String_equals(token, "EXEC")) {
    dptr->d_calltype = sidlx_rmi_CallType_EXEC;

    token = get_next_token(self, _ex); SIDL_CHECK(*_ex);
    if(!sidl_String_equals(token, "objid")) {
      SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.init:Improperly formed call!");  
    }
    
    token = get_next_token(self, _ex); SIDL_CHECK(*_ex);
    dptr->d_objid = sidl_String_strdup(token); /*This could be eliminated to save time*/


    token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
    if(!sidl_String_equals(token, "clsid")) {
      SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.init:Improperly formed response!");
    }
    
    token = get_next_token(self, _ex); SIDL_CHECK(*_ex);
    dptr->d_clsid = sidl_String_strdup(token); /*This could be eliminated to save time*/
    

    token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
    if(!sidl_String_equals(token, "method")) {
      SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.init:Improperly formed response!");
    }

    token = get_next_token(self, _ex); SIDL_CHECK(*_ex);
    dptr->d_methodName = sidl_String_strdup(token); /*This could be eliminated to save time*/

    token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
    if(!sidl_String_equals(token, "args")) {
      SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.init:Improperly formed response!");
    }
    /* Now return and the arguments will be unserialized by the ORB*/

  } else if(sidl_String_equals(token, "CONNECT")) {
    char * type = NULL;
    dptr->d_calltype = sidlx_rmi_CallType_CONNECT;
    sidlx_rmi_SimCall_unpackString(self, "objectID", &(dptr->d_objid), _ex); SIDL_CHECK(*_ex);
    dptr->d_methodName = sidl_String_strdup("CONNECT");
    sidlx_rmi_SimCall_unpackString(self, "className", &(dptr->d_clsid), _ex); SIDL_CHECK(*_ex);
    /* Now return and the connecteeURL will be unserialized by the ORB*/
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.init:Improperly formed response!");

  }

  return;
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.init) */
}

/*
 * Method:  getMethodName[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_getMethodName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimCall_getMethodName(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.getMethodName) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if (dptr) {
    return sidl_String_strdup(dptr->d_methodName);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This call has not been initialized yet.!");
  }
 EXIT:
  return NULL;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.getMethodName) */
}

/*
 * Method:  getObjectID[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_getObjectID"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimCall_getObjectID(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.getObjectID) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if (dptr) {
    return sidl_String_strdup(dptr->d_objid);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This call has not been initialized yet.!");
  }
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.getObjectID) */
}

/*
 * Method:  getClassName[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_getClassName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimCall_getClassName(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.getClassName) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if (dptr) {
    return sidl_String_strdup(dptr->d_clsid);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This call has not been initialized yet.!");
  }
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.getClassName) */
}

/*
 * Method:  getCallType[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_getCallType"

#ifdef __cplusplus
extern "C"
#endif
enum sidlx_rmi_CallType__enum
impl_sidlx_rmi_SimCall_getCallType(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.getCallType) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if (dptr) {
    return dptr->d_calltype;
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This call has not been initialized yet.!");
  }
 EXIT:
  return 0;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.getCallType) */
}

/*
 * Method:  unpackBool[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_unpackBool"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_unpackBool(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.unpackBool) */
    struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr) {
    char temp;
    unserialize(self, &temp, 1, _ex); SIDL_CHECK(*_ex);
    if(temp == 0) {
      *value = 0;  /*false*/
    }else {
      *value = 1;  /*true*/
    }
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This SimCall not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.unpackBool) */
}

/*
 * Method:  unpackChar[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_unpackChar"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_unpackChar(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.unpackChar) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr) {
    unserialize(self, value, 1, _ex); SIDL_CHECK(*_ex);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This SimCall not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.unpackChar) */
}

/*
 * Method:  unpackInt[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_unpackInt"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_unpackInt(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.unpackInt) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr) {
    int32_t temp;
    unserialize(self, (char*)&temp, 4, _ex); SIDL_CHECK(*_ex);
    *value = ntohl(temp);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This SimCall not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.unpackInt) */
}

/*
 * Method:  unpackLong[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_unpackLong"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_unpackLong(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ int64_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.unpackLong) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  short host = 1;
  short net = ntohs(host);
  if(dptr) {
    int64_t temp;
    unserialize(self, (char*)&temp, 8, _ex); SIDL_CHECK(*_ex);
    if(host == net) {  /*This computer uses network byte ordering*/
      *value = temp;
    } else {           /*This computer does not use network byte ordering*/
      *value = temp;
      flip64(value);
    }
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This SimCall not initilized!");  
  }
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.unpackLong) */
}

/*
 * Method:  unpackFloat[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_unpackFloat"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_unpackFloat(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ float* value,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.unpackFloat) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr) {
    short host = 1;
    short net = htons(host);
    if(host == net) {  /*This computer uses network byte ordering*/
      unserialize(self, (char*)value, 4, _ex); SIDL_CHECK(*_ex);
    } else {           /*This computer does not use network byte ordering*/
      unserialize(self, (char*)value, 4, _ex); SIDL_CHECK(*_ex);
      flip32((int32_t*)value);
    }
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This SimCall not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.unpackFloat) */
}

/*
 * Method:  unpackDouble[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_unpackDouble"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_unpackDouble(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.unpackDouble) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr) {
    short host = 1;
    short net = htons(host);
    if(host == net) {  /*This computer uses network byte ordering*/
      unserialize(self, (char*)value, 8, _ex); SIDL_CHECK(*_ex);
    } else {           /*This computer does not use network byte ordering*/
      unserialize(self, (char*)value, 8, _ex); SIDL_CHECK(*_ex);
      flip64((int64_t*)value);
    }

  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This SimCall not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.unpackDouble) */
}

/*
 * Method:  unpackFcomplex[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_unpackFcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_unpackFcomplex(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.unpackFcomplex) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr) {
    short host = 1;
    short net = htons(host);
    if(host == net) {  /*This computer uses network byte ordering*/
      unserialize(self, (char*)(&(value->real)), 4, _ex); SIDL_CHECK(*_ex);
      unserialize(self, (char*)(&(value->imaginary)), 4, _ex); SIDL_CHECK(*_ex);
    } else {           /*This computer does not use network byte ordering*/
      unserialize(self, (char*)(&(value->real)), 4, _ex); SIDL_CHECK(*_ex);
      unserialize(self, (char*)(&(value->imaginary)), 4, _ex); SIDL_CHECK(*_ex);
      flip32((int32_t*)&(value->real));
      flip32((int32_t*)&(value->imaginary));
    }

  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This SimCall not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.unpackFcomplex) */
}

/*
 * Method:  unpackDcomplex[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_unpackDcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_unpackDcomplex(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.unpackDcomplex) */
  struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr) {
    short host = 1;
    short net = htons(host);
    if(host == net) {  /*This computer uses network byte ordering*/
      unserialize(self, (char*)(&(value->real)), 8, _ex); SIDL_CHECK(*_ex);
      unserialize(self, (char*)(&(value->imaginary)), 8, _ex); SIDL_CHECK(*_ex);
    } else {           /*This computer does not use network byte ordering*/
      unserialize(self, (char*)(&(value->real)), 8, _ex); SIDL_CHECK(*_ex);
      unserialize(self, (char*)(&(value->imaginary)), 8, _ex); SIDL_CHECK(*_ex);
      flip64((int64_t*)&(value->real));
      flip64((int64_t*)&(value->imaginary));
    }

  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This SimCall not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.unpackDcomplex) */
}

/*
 * Method:  unpackString[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimCall_unpackString"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimCall_unpackString(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ char** value,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall.unpackString) */
    struct sidlx_rmi_SimCall__data *dptr =
    sidlx_rmi_SimCall__get_data(self);
  if(dptr) {
    int32_t temp = 0;
    int32_t len = 0;
    unserialize(self, (char*)&temp, 4, _ex); SIDL_CHECK(*_ex);
    len = ntohl(temp);
    *value = sidl_String_alloc(len);
    unserialize(self, *value, len, _ex); SIDL_CHECK(*_ex);
    (*value)[len] = '\0';
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimCall.getMethodName: This SimCall not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall.unpackString) */
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_SimCall_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_io_Deserializer__connect(url, _ex);
}
char * impl_sidlx_rmi_SimCall_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj) {
  return sidl_io_Deserializer__getURL(obj);
}
struct sidlx_rmi_SimCall__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimCall__connect(url, _ex);
}
char * impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_SimCall(struct 
  sidlx_rmi_SimCall__object* obj) {
  return sidlx_rmi_SimCall__getURL(obj);
}
struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_io_IOException__connect(url, _ex);
}
char * impl_sidlx_rmi_SimCall_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) {
  return sidl_io_IOException__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_SimCall_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
