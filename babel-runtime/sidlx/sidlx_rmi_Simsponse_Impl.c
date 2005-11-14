/*
 * File:          sidlx_rmi_Simsponse_Impl.c
 * Symbol:        sidlx.rmi.Simsponse-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.rmi.Simsponse
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.Simsponse" (version 0.1)
 * 
 * implementation of Response using the Simocol (simple-protocol), 
 * 	contains all the serialization code
 */

#include "sidlx_rmi_Simsponse_Impl.h"

#line 27 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._includes) */
#include "sidlx_rmi_GenNetworkException.h"
#include "sidlType.h"
#include "sidl_Exception.h"
#include "sidl_String.h"
#include <stdlib.h>
#include <string.h>
/** Parses string into tokens, replaces token seperator with '\0' and
 *  returns the pointer to the beginning of this token.  Should only be used
 *  when you know you're dealing with an alpha-numeric string.
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
	SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.get_next_token:Improperly formed response!");  
      }
    }
    *s_ptr = '\0';
    ++(dptr->d_current); /* Advance the the beginning of the next token */
    return begin;
  EXIT:
    return NULL;
  }
}

static void unserialize(sidlx_rmi_Simsponse self, char* data, int n, sidl_BaseInterface* _ex) {
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  int i = 0;
  char* d_buf = sidl_char__array_first(dptr->d_carray);
  int d_capacity = sidl_char__array_length(dptr->d_carray, 0);
  int rem = d_capacity - dptr->d_current; /*space remaining*/
  char* s_ptr =  s_ptr = (d_buf)+(dptr->d_current);
  if(n>rem) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.unserialize: Not enough data left!");  
  }
  memcpy(data, s_ptr,n);
  (dptr->d_current) += n;
 EXIT:
  return;
}

static void unserialize_exception(sidlx_rmi_Simsponse self, sidl_BaseInterface* _ex) {
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);

  /*TODO: Actually unserialize an exception here.  For now, just make a place holder*/
  sidlx_rmi_GenNetworkException temp_ex = sidlx_rmi_GenNetworkException__create();
  sidlx_rmi_GenNetworkException_setNote(temp_ex, "An exception was thrown from the remote method. This is a placeholder");
  dptr->d_exception = sidl_BaseException__cast(temp_ex);
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

/* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._includes) */
#line 113 "sidlx_rmi_Simsponse_Impl.c"

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
  void)
{
#line 127 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._load) */
  /* insert implementation here: sidlx.rmi.Simsponse._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._load) */
#line 133 "sidlx_rmi_Simsponse_Impl.c"
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
  /* in */ sidlx_rmi_Simsponse self)
{
#line 145 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._ctor) */
  /* insert implementation here: sidlx.rmi.Simsponse._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._ctor) */
#line 153 "sidlx_rmi_Simsponse_Impl.c"
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
  /* in */ sidlx_rmi_Simsponse self)
{
#line 164 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._dtor) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    sidlx_rmi_Socket_deleteRef(dptr->d_sock);
    sidl_char__array_deleteRef(dptr->d_carray);
    sidl_String_free((void*)dptr->d_className);
    sidl_String_free((void*)dptr->d_methodName);
    sidl_String_free((void*)dptr->d_objectID);
    if(dptr->d_exception) {
      sidl_BaseException_deleteRef( dptr->d_exception);
    }
    free((void*)dptr);
    sidlx_rmi_Simsponse__set_data(self, NULL);
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._dtor) */
#line 187 "sidlx_rmi_Simsponse_Impl.c"
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
  /* in */ const char* className,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex)
{
#line 201 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.init) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  char* token = NULL;
  if (dptr) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "This response has already been init'ed!");
  } else {
    dptr = malloc(sizeof(struct sidlx_rmi_Simsponse__data));
  }
  dptr->d_methodName = sidl_String_strdup(methodName);
  dptr->d_className = sidl_String_strdup(className);
  dptr->d_objectID = sidl_String_strdup(objectid);
  sidlx_rmi_Socket_addRef(sock);
  dptr->d_sock = sock;
  dptr->d_carray = NULL;
  dptr->d_exception = NULL;
  dptr->d_current = 0;
  sidlx_rmi_Simsponse__set_data(self, dptr);

  sidlx_rmi_Socket_readstring_alloc(sock,&(dptr->d_carray),_ex);SIDL_CHECK(*_ex);

  token = get_next_token(self, _ex); SIDL_CHECK(*_ex);
  if(!sidl_String_equals(token, "RESP")) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.init:Improperly formed response!");
  }

  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(!sidl_String_equals(token, "objid")) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.init:Improperly formed response!");
  }

  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(!dptr->d_objectID) { /*If this object was just created, we won't know the objectID yet*/  
    dptr->d_objectID = sidl_String_strdup(token);
  } else {
    if(!sidl_String_equals(token, objectid)) {
      SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.init:Response for the wrong object?!");
    }
  }
  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(!sidl_String_equals(token, "clsid")) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.init:Improperly formed response!");  
  }


  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(className) {
    if(!sidl_String_equals(token, className)) {
      SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.init:Object ID matches, but className is wrong!");
    }
  }

  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(!sidl_String_equals(token, "method")) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.init:Improperly formed response!");  
  }

  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(methodName) {
    if(!sidl_String_equals(token, methodName)) {
      SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.init:Object ID and clsss match, but methodName is wrong!");
    }
  }

 /* if args, do nothing (normal case), if exception, remote method threw an exception,  
  * unserialize it and set it in the object data. otherwise, we have a problem, throw 
  * real exception 
  */
  token = get_next_token(self, _ex);SIDL_CHECK(*_ex);
  if(!sidl_String_equals(token, "args")) {
    if(sidl_String_equals(token, "exception")) {
      unserialize_exception(self, _ex);
    } else {
      SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.init:Improperly formed response!"); }
  }
  


 EXIT:
  /*Not really much to do here...*/
    return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.init) */
#line 293 "sidlx_rmi_Simsponse_Impl.c"
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
#line 301 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.getMethodName) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    return sidl_String_strdup(dptr->d_methodName);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.getMethodName) */
#line 323 "sidlx_rmi_Simsponse_Impl.c"
}

/*
 * Method:  getClassName[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_getClassName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_Simsponse_getClassName(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 329 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.getClassName) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    return sidl_String_strdup(dptr->d_className);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.getClassName) */
#line 353 "sidlx_rmi_Simsponse_Impl.c"
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
#line 357 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.getObjectID) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    return sidl_String_strdup(dptr->d_objectID);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.getObjectID) */
#line 383 "sidlx_rmi_Simsponse_Impl.c"
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
#line 387 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackBool) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    char temp;
    unserialize(self, &temp, 1, _ex); SIDL_CHECK(*_ex);
    if(temp == 0) {
      *value = 0;  /*false*/
    }else {
      *value = 1;  /*true*/
    }
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackBool) */
#line 421 "sidlx_rmi_Simsponse_Impl.c"
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
#line 423 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackChar) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    unserialize(self, value, 1, _ex); SIDL_CHECK(*_ex);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackChar) */
#line 453 "sidlx_rmi_Simsponse_Impl.c"
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
#line 453 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackInt) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    int temp;
    unserialize(self, (char*)&temp, 4, _ex); SIDL_CHECK(*_ex);
    *value = ntohl(temp);
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackInt) */
#line 487 "sidlx_rmi_Simsponse_Impl.c"
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
#line 485 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
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
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackLong) */
#line 529 "sidlx_rmi_Simsponse_Impl.c"
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
#line 525 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackFloat) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
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
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackFloat) */
#line 568 "sidlx_rmi_Simsponse_Impl.c"
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
#line 562 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackDouble) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
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
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackDouble) */
#line 608 "sidlx_rmi_Simsponse_Impl.c"
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
#line 600 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackFcomplex) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
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
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackFcomplex) */
#line 651 "sidlx_rmi_Simsponse_Impl.c"
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
#line 641 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackDcomplex) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
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
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackDcomplex) */
#line 694 "sidlx_rmi_Simsponse_Impl.c"
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
#line 682 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.unpackString) */
    struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    int temp;
    int len;
    unserialize(self, (char*)&temp, 4, _ex); SIDL_CHECK(*_ex);
    len = ntohl(temp);
    *value = sidl_String_alloc(len);
    unserialize(self, *value, len, _ex); SIDL_CHECK(*_ex);
    (*value)[len] = '\0';
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  }
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.unpackString) */
#line 733 "sidlx_rmi_Simsponse_Impl.c"
}

/*
 * if returns null, then safe to unpack arguments 
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
#line 717 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.getExceptionThrown) */
  /*TODO: Where did this come from, what to do?*/
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr->d_exception) {
    sidl_BaseException_addRef(dptr->d_exception);
    return dptr->d_exception;
  } 
  return NULL; 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.getExceptionThrown) */
#line 762 "sidlx_rmi_Simsponse_Impl.c"
}

/*
 * signal that all is complete 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simsponse_done"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_rmi_Simsponse_done(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 744 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse.done) */
  struct sidlx_rmi_Simsponse__data *dptr =
    sidlx_rmi_Simsponse__get_data(self);
  if(dptr) {
    if(dptr->d_current == sidl_char__array_length(dptr->d_carray,0)) {
      return 1;
    } else {
      return 0;
    }
  } 
  SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simsponse.getMethodName: This Simsponse not initilized!");  
  
 EXIT:
  return 0;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse.done) */
#line 796 "sidlx_rmi_Simsponse_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_Response__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj) {
  return sidl_rmi_Response__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_io_Deserializer__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj) {
  return sidl_io_Deserializer__getURL(obj);
}
struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Simsponse__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Simsponse(struct 
  sidlx_rmi_Simsponse__object* obj) {
  return sidlx_rmi_Simsponse__getURL(obj);
}
struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_io_IOException__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) {
  return sidl_io_IOException__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseException__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj) {
  return sidl_BaseException__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
