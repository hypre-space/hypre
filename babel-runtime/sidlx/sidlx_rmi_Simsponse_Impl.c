/*
 * File:          sidlx_rmi_Simsponse_Impl.c
 * Symbol:        sidlx.rmi.Simsponse-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for sidlx.rmi.Simsponse
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.Simsponse" (version 0.1)
 * 
 * implementation of Response using the Simhandle Protocol (written by Jim)
 */

#include "sidlx_rmi_Simsponse_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._includes) */
#include "sidl_rmi_NetworkException.h"
#include "sidlType.h"
#include "sidl_Exception.h"
#include "sidl_String.h"
#include "sidl_DLL.h"
#include "sidl_Resolve.h"
#include "sidl_Scope.h"
#include "sidl_Loader.h"
#include "sidl_io_Serializable.h"
#include "sidl_rmi_ProtocolFactory.h"
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
/** Parses string into tokens, replaces token seperator with '\0' and
 *  returns the pointer to the beginning of this token.  Should only be used
 *  when you know you're dealing with an alpha-numeric string.
 *  It assume the seperator is ':' as dictated by our protocol 
 */

static char* get_next_token(sidlx_rmi_Simsponse self,/*out*/ sidl_BaseInterface* _ex) {
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr){
    /*   int counter = dptr->d_current; */
    int upper = sidl_char__array_upper(dptr->d_carray,0);
    char* d_buf = sidl_char__array_first(dptr->d_carray);
    char* begin = d_buf+dptr->d_current;
    char* s_ptr = begin;

    while(*s_ptr != ':') {
      ++s_ptr;
      ++(dptr->d_current);
      if(*s_ptr == '\0' || dptr->d_current > upper) {
	SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.get_next_token:Improperly formed response!");  
      }
    }
    *s_ptr = '\0';
    ++(dptr->d_current); /* Advance the the beginning of the next token */
    return begin;
  EXIT:
    return NULL;
  }
  return NULL;
}

/* Copies n bytes from the buffer into data*/
static void unserialize(sidlx_rmi_Simsponse self, char* data, int n, sidl_BaseInterface* _ex) {
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  char* d_buf = sidl_char__array_first(dptr->d_carray);
  int d_capacity = sidl_char__array_length(dptr->d_carray, 0);
  int rem = d_capacity - dptr->d_current; /*space remaining*/
  char* s_ptr =  s_ptr = (d_buf)+(dptr->d_current);
  if(n>rem) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.unserialize: Not enough data left!");  
  }
  memcpy(data, s_ptr,n);
  (dptr->d_current) += n;
 EXIT:
  return;
}
/* Moves some pointers around to pass back the chunk of space that should be
   holding array data. total_len is the expected length of the array data
   IN BYTES*/
static void* buffer_array(sidlx_rmi_Simsponse self, int64_t total_len, sidl_BaseInterface* _ex) {
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  char* d_buf = sidl_char__array_first(dptr->d_carray);
  int d_capacity = sidl_char__array_length(dptr->d_carray, 0);
  int rem = d_capacity - dptr->d_current; /*space remaining*/
  char* s_ptr =  s_ptr = (d_buf)+(dptr->d_current);
  if(total_len>rem) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.unserialize: Not enough data left!");  
  }
  (dptr->d_current) += total_len;
  
  return s_ptr;
 EXIT:
  return NULL;
}

/* Checks dimension and array boundries to make sure they match*/
static sidl_bool check_bounds(struct sidl__array* a, int32_t dimen, int32_t* lower, int32_t* upper) {
  int32_t i;
  if(a && sidlArrayDim(a) == dimen) {
    for(i = 0; i < dimen; ++i) {
      if(sidlLower(a,i) != lower[i] || sidlUpper(a,i) != upper[i]) {
	return FALSE;
      }
    }
    return TRUE;
  }
  return FALSE;
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

/* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._load) */
  /* insert implementation here: sidlx.rmi.Simsponse._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse__ctor(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._ctor) */
  struct sidlx_rmi_Simsponse__data *dptr = (struct sidlx_rmi_Simsponse__data *)
    malloc(sizeof(struct sidlx_rmi_Simsponse__data));
  dptr->d_methodName = NULL;
  dptr->d_objectID = NULL;
  dptr->d_sock = NULL;
  dptr->d_carray = NULL;
  dptr->d_exception = NULL;
  dptr->d_current = 0;
  sidlx_rmi_Simsponse__set_data(self, dptr);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse__ctor2(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.Simsponse._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse__dtor(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._dtor) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    sidlx_rmi_Socket_deleteRef(dptr->d_sock,_ex); SIDL_CHECK(*_ex);
    sidl_char__array_deleteRef(dptr->d_carray);
    /*sidl_String_free((void*)dptr->d_className);*/
    sidl_String_free((void*)dptr->d_methodName);
    sidl_String_free((void*)dptr->d_objectID);
    if(dptr->d_exception) {
      sidl_BaseException_deleteRef(dptr->d_exception,_ex); SIDL_CHECK(*_ex);
    }
  EXIT:
    free((void*)dptr);
    sidlx_rmi_Simsponse__set_data(self, NULL);
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._dtor) */
  }
}

/*
 * Method:  init[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_init"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_init(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* methodName,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.init) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if (!dptr) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "This response has already been init'ed!");
  } 
  dptr->d_methodName = sidl_String_strdup(methodName);
  /*dptr->d_className = sidl_String_strdup(className);*/
  dptr->d_objectID = sidl_String_strdup(objectid);
  sidlx_rmi_Socket_addRef(sock,_ex); SIDL_CHECK(*_ex);
  dptr->d_sock = sock;

 EXIT:
  /*Not really much to do here...*/
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.init) */
  }
}

/*
 * Method:  test[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_test"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_rmi_Simsponse_test(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ int32_t secs,
  /* in */ int32_t usecs,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.test) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if ( dptr  && dptr->d_sock ) { 
    return sidlx_rmi_Socket_test(dptr->d_sock, secs, usecs, _ex);
  }
  return FALSE;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.test) */
  }
}

/*
 * Method:  pullData[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_pullData"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_pullData(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.pullData) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  char* token = NULL;
  sidl_bool _ex_thrown;
  sidl_io_Serializable ex_tmp = NULL;

  /* Allocate buffer space for the incoming data and copy it into the buffer*/
  sidlx_rmi_Socket_readstring_alloc(dptr->d_sock,&(dptr->d_carray),_ex);SIDL_CHECK(*_ex);

  /* Responses are of the format::
     RESP:objid:(objectID):method:(methodname):args:(arguments)
  */ 
  token = get_next_token(self, _ex); SIDL_CHECK(*_ex);
  if(!sidl_String_equals(token, "RESP")) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.init:Improperly formed response!");
  }

  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(!sidl_String_equals(token, "objid")) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.init:Improperly formed response!");
  }

  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(!dptr->d_objectID) { /*If this object was just created, we won't know the objectID yet*/  
    dptr->d_objectID = sidl_String_strdup(token);
  } else {
    if(!sidl_String_equals(token, dptr->d_objectID)) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.init:Response for the wrong object?!");
    }
  }
  /* We no longer transfer classnames with methodcalls*/
  /*token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
    if(!sidl_String_equals(token, "clsid")) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.init:Improperly formed response!");  
    }
    token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
    if(className) {
    if(!sidl_String_equals(token, className)) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.init:Object ID matches, but className is wrong!");
    }
    }
  */
  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(!sidl_String_equals(token, "method")) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.init:Improperly formed response!");  
  }

  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(dptr->d_methodName) {
    if(!sidl_String_equals(token, dptr->d_methodName)) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.init:Object ID and clsss match, but methodName is wrong!");
    }
  }

  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(!sidl_String_equals(token, "args")) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.init:Improperly formed response!"); 
  }
  
  sidlx_rmi_Simsponse_unpackBool(self, "_ex_thrown", &_ex_thrown, _ex); SIDL_CHECK(*_ex);
  if(_ex_thrown) {
    sidlx_rmi_Simsponse_unpackSerializable(self, "_ex", &ex_tmp, _ex); SIDL_CHECK(*_ex);
    dptr->d_exception = sidl_BaseException__cast(ex_tmp, _ex); SIDL_CHECK(*_ex);
    sidl_io_Serializable_deleteRef(ex_tmp, _ex); SIDL_CHECK(*_ex);
  }
  
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.pullData) */
  }
}

/*
 * Method:  getMethodName[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_getMethodName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_Simsponse_getMethodName(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.getMethodName) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    return sidl_String_strdup(dptr->d_methodName);
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.getMethodName) */
  }
}

/*
 * Method:  getObjectID[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_getObjectID"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_Simsponse_getObjectID(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.getObjectID) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    return sidl_String_strdup(dptr->d_objectID);
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.getObjectID) */
  }
}

/*
 *  
 * May return a communication exception or an execption thrown
 * from the remote server.  If it returns null, then it's safe
 * to unpack arguments
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_getExceptionThrown"

#ifdef __cplusplus
extern "C"
#endif
sidl_BaseException
impl_sidlx_rmi_Simsponse_getExceptionThrown(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.getExceptionThrown) */

  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr && dptr->d_exception) {
    sidl_BaseException_addRef(dptr->d_exception, _ex); SIDL_CHECK(*_ex);
    return dptr->d_exception;
  } 
 EXIT:
  return NULL; 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.getExceptionThrown) */
  }
}

/*
 * Method:  unpackBool[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackBool"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackBool(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackBool) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    char temp;
    unserialize(self, &temp, 1, _ex); SIDL_CHECK(*_ex);
    if(temp == 0) {
      *value = FALSE;  /*false*/
    }else {
      *value = TRUE;  /*true*/
    }
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackBool) */
  }
}

/*
 * Method:  unpackChar[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackChar"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackChar(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackChar) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    unserialize(self, value, 1, _ex); SIDL_CHECK(*_ex);
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackChar) */
  }
}

/*
 * Method:  unpackInt[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackInt"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackInt(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackInt) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    int temp;
    unserialize(self, (char*)&temp, 4, _ex); SIDL_CHECK(*_ex);
    *value = ntohl(temp);
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackInt) */
  }
}

/*
 * Method:  unpackLong[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackLong"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackLong(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ int64_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackLong) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
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
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackLong) */
  }
}

/*
 * Method:  unpackOpaque[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackOpaque"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackOpaque(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ void** value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackOpaque) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  short host = 1;
  short net = ntohs(host);
  if(dptr) {
    int64_t temp;
    unserialize(self, (char*)&temp, 8, _ex); SIDL_CHECK(*_ex);
    if(host == net) {  /*This computer uses network byte ordering*/
      *value = (void*) (ptrdiff_t) temp;
    } else {           /*This computer does not use network byte ordering*/
      flip64(&temp);
      *value = (void*) (ptrdiff_t) temp;
    }
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackOpaque) */
  }
}

/*
 * Method:  unpackFloat[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackFloat"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackFloat(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ float* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackFloat) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    unserialize(self, (char*)value, 4, _ex); SIDL_CHECK(*_ex);
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackFloat) */
  }
}

/*
 * Method:  unpackDouble[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackDouble"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackDouble(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackDouble) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    unserialize(self, (char*)value, 8, _ex); SIDL_CHECK(*_ex);
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackDouble) */
  }
}

/*
 * Method:  unpackFcomplex[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackFcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackFcomplex(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackFcomplex) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    unserialize(self, (char*)(&(value->real)), 4, _ex); SIDL_CHECK(*_ex);
    unserialize(self, (char*)(&(value->imaginary)), 4, _ex); SIDL_CHECK(*_ex);
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackFcomplex) */
  }
}

/*
 * Method:  unpackDcomplex[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackDcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackDcomplex(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackDcomplex) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    unserialize(self, (char*)(&(value->real)), 8, _ex); SIDL_CHECK(*_ex);
    unserialize(self, (char*)(&(value->imaginary)), 8, _ex); SIDL_CHECK(*_ex);
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackDcomplex) */
  }
}

/*
 * Method:  unpackString[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackString"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackString(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ char** value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackString) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    int temp;
    int len;
    unserialize(self, (char*)&temp, 4, _ex); SIDL_CHECK(*_ex);
    len = ntohl(temp);
    if(len <= 0) {
      *value = NULL;
      return;
    }
    *value = sidl_String_alloc(len);
    unserialize(self, *value, len, _ex); SIDL_CHECK(*_ex);
    (*value)[len] = '\0';
  } else {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackString) */
  }
}

/*
 * Method:  unpackSerializable[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackSerializable"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackSerializable(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ sidl_io_Serializable* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackSerializable) */
  char* className = NULL;
  sidl_DLL dll = NULL;
  sidl_BaseClass h = NULL;
  sidl_io_Deserializer ds = NULL;
  int is_remote = 0;
  char* obj_url = NULL;
  sidl_BaseInterface _throwaway_exception = NULL;

  sidlx_rmi_Simsponse_unpackBool(self, NULL, &is_remote, _ex); SIDL_CHECK(*_ex);
  if(is_remote) {
    sidlx_rmi_Simsponse_unpackString(self, NULL, &obj_url, _ex); SIDL_CHECK(*_ex); 
    if(obj_url == NULL) { *value = NULL; goto EXIT; }
    *value = sidl_rmi_ProtocolFactory_unserializeInstance(obj_url, _ex); SIDL_CHECK(*_ex);
    
  } else {
    ds = sidl_io_Deserializer__cast(self, _ex); SIDL_CHECK(*_ex);
    sidl_io_Deserializer_unpackString(ds, NULL, &className, _ex); SIDL_CHECK(*_ex); 
    dll = sidl_Loader_findLibrary(className, "ior/impl", sidl_Scope_SCLSCOPE, sidl_Resolve_SCLRESOLVE, _ex); SIDL_CHECK(*_ex);
    h = sidl_DLL_createClass(dll, className, _ex); SIDL_CHECK(*_ex);
    *value = sidl_io_Serializable__cast(h, _ex); SIDL_CHECK(*_ex);
    /*ds = sidl_io_Deserializer__cast(self, _ex); SIDL_CHECK(*_ex); */
    sidl_io_Serializable_unpackObj(*value, ds, _ex); SIDL_CHECK(*_ex);
  }

 EXIT:
  if (ds) { sidl_io_Deserializer_deleteRef(ds, &_throwaway_exception); }
  if (dll){ sidl_DLL_deleteRef(dll, &_throwaway_exception); }
  if (h) {sidl_BaseClass_deleteRef(h, &_throwaway_exception); }
  sidl_String_free(obj_url);
  sidl_String_free(className);

  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackSerializable) */
  }
}

/*
 *  unpack arrays of values 
 * It is possible to ensure an array is
 * in a certain order by passing in ordering and dimension
 * requirements.  ordering should represent a value in the
 * sidl_array_ordering enumeration in sidlArray.h If either
 * argument is 0, it means there is no restriction on that
 * aspect.  The rarray flag should be set if the array being
 * passed in is actually an rarray.  The semantics are slightly
 * different for rarrays.  The passed in array MUST be reused,
 * even if the array has changed bounds.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackBoolArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackBoolArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<bool> */ struct sidl_bool__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackBoolArray) */
  sidl_bool isRow;
  char* srcFirst = NULL;
  sidl_bool* destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t *src_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  sidl_bool reuse = FALSE;

  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }

  /* If we want to reuse the array, and the array is reuseable, we don't have to
   * do anything here.  Otherwise, we need to either create a new array, or throw 
   * an exception.  (It shouldn't be possible that an rarray could not 
   * be reuseable
   */
  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_bool__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_bool__array_createCol(l_dimen,lower,upper);
      }
    }
  }

 
  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginnings of both arrays*/
  srcFirst = buffer_array(self, t_len*1, _ex);SIDL_CHECK(*_ex);
  destFirst = sidl_bool__array_first(*value);

  dest_stride = (*value)->d_metadata.d_stride;
  src_stride = dest_stride; /*SHOULD be the same, figured out remotely*/
  if(t_len > 0) {
    do {
      if(*srcFirst == 0) {
	*destFirst = FALSE;
      } else {
	*destFirst = TRUE;
      }
      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackBoolArray) */
  }
}

/*
 * Method:  unpackCharArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackCharArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackCharArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<char> */ struct sidl_char__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackCharArray) */
  sidl_bool isRow;
  char* srcFirst = NULL;
  char* destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t *src_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  sidl_bool reuse = FALSE;
  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }
  /* If we want to reuse the array, and the array is reuseable, we don't have to
   * do anything here.  Otherwise, we need to either create a new array, or throw 
   * an exception.  (It shouldn't be possible that an rarray could not 
   * be reuseable
   */
  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_char__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_char__array_createCol(l_dimen,lower,upper);
      }
    }
  }

  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginnings of both arrays*/
  srcFirst = buffer_array(self, t_len*1, _ex);SIDL_CHECK(*_ex);
  destFirst = sidl_char__array_first(*value);

  dest_stride = (*value)->d_metadata.d_stride;
  src_stride = dest_stride; /*SHOULD be the same, figured out remotely*/
  if(t_len > 0) {
    do {
      *destFirst = *srcFirst;
      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackCharArray) */
  }
}

/*
 * Method:  unpackIntArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackIntArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackIntArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<int> */ struct sidl_int__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackIntArray) */
  sidl_bool isRow;
  int32_t* srcFirst = NULL;
  int32_t* destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t *src_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  sidl_bool reuse = FALSE;

  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }

  /* If we want to reuse the array, and the array is reuseable, we don't have to
   * do anything here.  Otherwise, we need to either create a new array, or throw 
   * an exception.  (It shouldn't be possible that an rarray could not 
   * be reuseable
   */
  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_int__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_int__array_createCol(l_dimen,lower,upper);
      }
    }
  }

  /* Create the destination array*/
  /*  if(isRow) {
   *value = sidl_int__array_createRow(l_dimen,lower,upper);
   } else {
   *value = sidl_int__array_createCol(l_dimen,lower,upper);
   }*/
  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginnings of both arrays*/
  srcFirst = buffer_array(self, t_len*4, _ex);SIDL_CHECK(*_ex);
  destFirst = sidl_int__array_first(*value);

  dest_stride = (*value)->d_metadata.d_stride;
  src_stride = dest_stride; /*SHOULD be the same, figured out remotely*/
  if(t_len > 0) {
    do {
      *destFirst = ntohl(*srcFirst);
      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackIntArray) */
  }
}

/*
 * Method:  unpackLongArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackLongArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackLongArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<long> */ struct sidl_long__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackLongArray) */
  sidl_bool isRow;
  int64_t* srcFirst = NULL;
  int64_t* destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t *src_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  short host = 1;
  short net = htons(host);
  sidl_bool reuse = FALSE;

  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }
  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_long__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_long__array_createCol(l_dimen,lower,upper);
      }
    }
  }
  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginnings of both arrays*/
  srcFirst = buffer_array(self, t_len*8, _ex);SIDL_CHECK(*_ex);
  destFirst = sidl_long__array_first(*value);

  dest_stride = (*value)->d_metadata.d_stride;
  src_stride = dest_stride; /*SHOULD be the same, figured out remotely*/
  if(t_len > 0) {
    do {
      if(host == net) {  /*This computer uses network byte ordering*/
	*destFirst = *srcFirst;
      } else {           /*This computer does not use network byte ordering*/
	int64_t tmp = *srcFirst;
	flip64((int64_t*)&tmp);
	*destFirst = tmp;
      }

      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackLongArray) */
  }
}

/*
 * Method:  unpackOpaqueArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackOpaqueArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackOpaqueArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<opaque> */ struct sidl_opaque__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackOpaqueArray) */
  sidl_bool isRow;
  int64_t* srcFirst = NULL;
  void** destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t *src_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  short host = 1;
  short net = htons(host);
  sidl_bool reuse = FALSE;
  int64_t temp = 0;

  /*Unserialize isRolw and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }
  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value) )) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_opaque__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_opaque__array_createCol(l_dimen,lower,upper);
      }
    }
  }
  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginnings of both arrays*/
  srcFirst = buffer_array(self, t_len*8, _ex);SIDL_CHECK(*_ex);
  destFirst = sidl_opaque__array_first(*value);

  dest_stride = (*value)->d_metadata.d_stride;
  src_stride = dest_stride; /*SHOULD be the same, figured out remotely*/
  if(t_len > 0) {
    do {
      if(host == net) {  /*This computer uses network byte ordering*/
	temp = *srcFirst;
	*destFirst = (void*) (ptrdiff_t)temp;
      } else {           /*This computer does not use network byte ordering*/
	temp = *srcFirst;
	flip64((int64_t*)&temp);
	*destFirst = (void*) (ptrdiff_t) temp;
      }

      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackOpaqueArray) */
  }
}

/*
 * Method:  unpackFloatArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackFloatArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackFloatArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<float> */ struct sidl_float__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackFloatArray) */
  sidl_bool isRow;
  float* srcFirst = NULL;
  float* destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t *src_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  sidl_bool reuse = FALSE;

  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }
  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_float__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_float__array_createCol(l_dimen,lower,upper);
      }
    }
  }

  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginnings of both arrays*/
  srcFirst = buffer_array(self, t_len*4, _ex);SIDL_CHECK(*_ex);
  destFirst = sidl_float__array_first(*value);

  dest_stride = (*value)->d_metadata.d_stride;
  src_stride = dest_stride; /*SHOULD be the same, figured out remotely*/
  if(t_len > 0) {
    do {
      *destFirst = *srcFirst;

      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackFloatArray) */
  }
}

/*
 * Method:  unpackDoubleArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackDoubleArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackDoubleArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<double> */ struct sidl_double__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackDoubleArray) */
  sidl_bool isRow;
  double* srcFirst = NULL;
  double* destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t *src_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  sidl_bool reuse = FALSE;

  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }

  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_double__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_double__array_createCol(l_dimen,lower,upper);
      }
    }
  }

  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginnings of both arrays*/
  srcFirst = (double*)buffer_array(self, t_len*8, _ex);SIDL_CHECK(*_ex);
  destFirst = sidl_double__array_first(*value);

  dest_stride = (*value)->d_metadata.d_stride;
  src_stride = dest_stride; /*SHOULD be the same, figured out remotely*/
  if(t_len > 0) {
    do {
      *destFirst = *srcFirst;

      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;
  
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackDoubleArray) */
  }
}

/*
 * Method:  unpackFcomplexArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackFcomplexArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackFcomplexArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<fcomplex> */ struct sidl_fcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackFcomplexArray) */
  sidl_bool isRow;
  struct sidl_fcomplex* srcFirst = NULL;
  struct sidl_fcomplex* destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t *src_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  sidl_bool reuse = FALSE;

  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }

  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_fcomplex__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_fcomplex__array_createCol(l_dimen,lower,upper);
      }
    }
  }

  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginnings of both arrays*/
  srcFirst = buffer_array(self, t_len*8, _ex);SIDL_CHECK(*_ex);
  destFirst = sidl_fcomplex__array_first(*value);

  dest_stride = (*value)->d_metadata.d_stride;
  src_stride = dest_stride; /*SHOULD be the same, figured out remotely*/
  if(t_len > 0) {
    do {
      destFirst->real = srcFirst->real;
      destFirst->imaginary = srcFirst->imaginary;
    
      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;
  
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackFcomplexArray) */
  }
}

/*
 * Method:  unpackDcomplexArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackDcomplexArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackDcomplexArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<dcomplex> */ struct sidl_dcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackDcomplexArray) */
  sidl_bool isRow;
  struct sidl_dcomplex* srcFirst = NULL;
  struct sidl_dcomplex* destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t *src_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  sidl_bool reuse = FALSE;

  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }
  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_dcomplex__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_dcomplex__array_createCol(l_dimen,lower,upper);
      }
    }
  }

  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginnings of both arrays*/
  srcFirst = buffer_array(self, t_len*16, _ex);SIDL_CHECK(*_ex);
  destFirst = sidl_dcomplex__array_first(*value);

  dest_stride = (*value)->d_metadata.d_stride;
  src_stride = dest_stride; /*SHOULD be the same, figured out remotely*/
  if(t_len > 0) {
    do {
      destFirst->real = srcFirst->real;
      destFirst->imaginary = srcFirst->imaginary;
    
      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;
 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackDcomplexArray) */
  }
}

/*
 * Method:  unpackStringArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackStringArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackStringArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<string> */ struct sidl_string__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackStringArray) */
  sidl_bool isRow;
  char** destFirst = NULL;
  int32_t l_dimen = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *dest_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  sidl_bool reuse = FALSE;
  
  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }

  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_string__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_string__array_createCol(l_dimen,lower,upper);
      }
    }
  }

  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginning the dest array*/
  /*HACK*/
  destFirst = (char**)sidl_int__array_first((struct sidl_int__array*) *value);

  dest_stride = ((struct sidl__array*)(*value))->d_stride;
  if(t_len > 0) {
    do {
      int temp;
      int len;
      unserialize(self, (char*)&temp, 4, _ex); SIDL_CHECK(*_ex);
      len = ntohl(temp);
      if(len <= 0) {
	*destFirst = NULL;
      } else {
	*destFirst = sidl_String_alloc(len);
	unserialize(self, *destFirst, len, _ex); SIDL_CHECK(*_ex);
	(*destFirst)[len] = '\0';
      }
      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackStringArray) */
  }
}

/*
 * Method:  unpackGenericArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackGenericArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackGenericArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<> */ struct sidl__array** value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackGenericArray) */

  int32_t type = 0;
  /*serialize type */
  sidlx_rmi_Simsponse_unpackInt(self, NULL, &type, _ex); SIDL_CHECK(*_ex);
  if(type == 0) {
    /* If array is null, return null array*/
    *value = NULL;
    return;
  }
  switch(type) {
  case sidl_bool_array:
    sidlx_rmi_Simsponse_unpackBoolArray(self,key,(struct sidl_bool__array**) value, 
					0,0,0,_ex);
    break;
  case sidl_char_array:
    sidlx_rmi_Simsponse_unpackCharArray(self,key,(struct sidl_char__array**) value, 
					0,0,0,_ex);
    break;
  case sidl_dcomplex_array:
    sidlx_rmi_Simsponse_unpackDcomplexArray(self,key,(struct sidl_dcomplex__array**) value, 
					    0,0,0,_ex);
    break;
  case sidl_double_array:
    sidlx_rmi_Simsponse_unpackDoubleArray(self,key,(struct sidl_double__array**) value, 
					  0,0,0,_ex);
    break;
  case sidl_fcomplex_array:
    sidlx_rmi_Simsponse_unpackFcomplexArray(self,key,(struct sidl_fcomplex__array**) value, 
					    0,0,0,_ex);
    break;
  case sidl_float_array:
    sidlx_rmi_Simsponse_unpackFloatArray(self,key,(struct sidl_float__array**) value, 
					 0,0,0,_ex);
    break;
  case sidl_int_array:
    sidlx_rmi_Simsponse_unpackIntArray(self,key,(struct sidl_int__array**) value, 
				       0,0,0,_ex);
    break;
  case sidl_long_array:
    sidlx_rmi_Simsponse_unpackLongArray(self,key,(struct sidl_long__array**) value, 
					0,0,0,_ex);
    break;
  case sidl_opaque_array:
    sidlx_rmi_Simsponse_unpackOpaqueArray(self,key,(struct sidl_opaque__array**) value, 
					  0,0,0,_ex);
    break;
  case sidl_string_array:
    sidlx_rmi_Simsponse_unpackStringArray(self,key,(struct sidl_string__array**) value, 
					  0,0,0,_ex);
    break;
  case sidl_interface_array:
    sidlx_rmi_Simsponse_unpackSerializableArray(self,key,(struct sidl_io_Serializable__array**) value, 
						0,0,0,_ex);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackGenericArray) */
  }
}

/*
 * Method:  unpackSerializableArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_unpackSerializableArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simsponse_unpackSerializableArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<sidl.io.Serializable> */ struct sidl_io_Serializable__array** 
    value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackSerializableArray) */
  sidl_bool isRow;
  sidl_io_Serializable* destFirst = NULL;
  int32_t l_dimen = 0;
  int32_t count = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t *dest_stride = NULL;
  int32_t lengths[7];
  int32_t current[7];
  int32_t lower[7];
  int32_t upper[7];
  int i;
  sidl_bool reuse = FALSE;
  
  /*Unserialize isRow and dimension*/
  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &isRow, _ex);  SIDL_CHECK(*_ex);
  impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, &l_dimen, _ex); SIDL_CHECK(*_ex);
  if(l_dimen == 0) {
    *value = NULL;  /*A zero dimension means a null array*/
    return;
  }

  impl_sidlx_rmi_Simsponse_unpackBool(self, NULL, &reuse, _ex);  SIDL_CHECK(*_ex);
  /*Unserialize arrays of upper and lower bounds*/
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, lower+count, _ex);
    SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_Simsponse_unpackInt(self, NULL, upper+count, _ex);
    SIDL_CHECK(*_ex);
  }

  if(!(reuse && check_bounds((struct sidl__array*)*value, l_dimen, lower, upper)
       && isRow == sidl__array_isRowOrder((struct sidl__array*)*value))) {
    if(isRarray && reuse) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Rarray has illeagally changed bounds remotely");
    } else {
      if(reuse && *value) {
	sidl__array_deleteRef((struct sidl__array*)*value);
      }
      /* Create the destination array*/
      if(isRow) {
	*value = sidl_io_Serializable__array_createRow(l_dimen,lower,upper);
      } else {
	*value = sidl_io_Serializable__array_createCol(l_dimen,lower,upper);
      }
    }
  }

  /* Figure out the lengths of each dimension, and total length*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(*value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*Get the pointers to the beginning the dest array*/
  /*HACK*/
  destFirst = (sidl_io_Serializable*)sidl_int__array_first((struct sidl_int__array*) *value);

  dest_stride = ((struct sidl__array*)(*value))->d_stride;
  if(t_len > 0) {
    do {
      sidlx_rmi_Simsponse_unpackSerializable(self, NULL, destFirst, _ex);
      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  destFirst -= ((lengths[i]-1) * dest_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  destFirst += dest_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackSerializableArray) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseException__connectI(url, ar, _ex);
}
struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseException__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Deserializer__connectI(url, ar, _ex);
}
struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_Deserializer__cast(bi, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializable__connectI(url, ar, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializable__cast(bi, _ex);
}
struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_Response__connectI(url, ar, _ex);
}
struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_rmi_Response(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_rmi_Response__cast(bi, _ex);
}
struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_Simsponse__connectI(url, ar, _ex);
}
struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidlx_rmi_Simsponse(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_Simsponse__cast(bi, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_Socket__cast(bi, _ex);
}
