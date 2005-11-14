/*
 * File:          sidlx_rmi_Simvocation_Impl.c
 * Symbol:        sidlx.rmi.Simvocation-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.rmi.Simvocation
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
 * Symbol "sidlx.rmi.Simvocation" (version 0.1)
 * 
 * implementation of Invocation using the Simocol (simple-protocol), 
 * 	contains all the serialization code
 */

#include "sidlx_rmi_Simvocation_Impl.h"

#line 27 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation._includes) */

#include <stdlib.h>
#include "sidl_String.h"
#include "sidl_Exception.h"
#include "sidlx_rmi_Simsponse.h"
#include "sidlx_rmi_GenNetworkException.h"
#include <string.h>
/* Copies n elements from data into the vector we maintain as part of this 
   Simvocation object.  This function will realloc as nessecary */
static void serialize(sidlx_rmi_Simvocation self, const char* data, int n, sidl_BaseInterface* _ex) {
  struct sidlx_rmi_Simvocation__data *dptr =
    sidlx_rmi_Simvocation__get_data(self);
  int i = 0;
  int rem = dptr->d_capacity - dptr->d_len; /*space remaining*/
  char* s_ptr = NULL; 

  if(rem < n) {
    dptr->d_capacity *= 2;
    dptr->d_buf = (char*)realloc((void*)dptr->d_buf, dptr->d_capacity);
  }
  s_ptr = (dptr->d_buf)+(dptr->d_len);
  memcpy(s_ptr, data, n);
  (dptr->d_len) += n;
  sidlx_rmi_Simvocation__set_data(self, dptr);
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

/* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation._includes) */
#line 74 "sidlx_rmi_Simvocation_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation__load(
  void)
{
#line 88 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation._load) */
  /* insert implementation here: sidlx.rmi.Simvocation._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation._load) */
#line 94 "sidlx_rmi_Simvocation_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation__ctor(
  /* in */ sidlx_rmi_Simvocation self)
{
#line 106 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation._ctor) */
  /* insert implementation here: sidlx.rmi.Simvocation._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation._ctor) */
#line 114 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation__dtor(
  /* in */ sidlx_rmi_Simvocation self)
{
#line 125 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation._dtor) */
  struct sidlx_rmi_Simvocation__data *dptr =
    sidlx_rmi_Simvocation__get_data(self);
  if(dptr) {
    sidlx_rmi_Socket_deleteRef(dptr->d_sock);
    sidl_String_free((void*)dptr->d_buf);
    sidl_String_free((void*)dptr->d_methodName);
    sidl_String_free((void*)dptr->d_className);
    sidl_String_free((void*)dptr->d_objectID);
    free((void*)dptr);
    sidlx_rmi_Simvocation__set_data(self, NULL);
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation._dtor) */
#line 145 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  init[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_init"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_init(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* methodName,
  /* in */ const char* className,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex)
{
#line 159 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.init) */
  struct sidlx_rmi_Simvocation__data *dptr =
    sidlx_rmi_Simvocation__get_data(self);
  
  int m_len = sidl_String_strlen(methodName);
  int c_len = sidl_String_strlen(className);
  int o_len = sidl_String_strlen(objectid);
  int h_len = 5+6+o_len+7+c_len+8+m_len+6;  /*header length*/
  int t_capacity = h_len+128;
  /* Make this inital space for the function call equal the length of
   * EXEC:objid:<objectid>:clsid:<className>:method:<methodName>:args:(args go here)
   * Make the space for args be equal to 128 bytes (just a guess) expands to size
   */
  

  if (dptr) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "This invocation has already been init'ed!");
  } else {
    dptr = malloc(sizeof(struct sidlx_rmi_Simvocation__data));
  }
  dptr->d_methodName = sidl_String_strdup(methodName);
  sidlx_rmi_Socket_addRef(sock);
  dptr->d_sock = sock;
  dptr->d_len = 0;
  dptr->d_capacity = t_capacity;
  dptr->d_buf = (char*)malloc(t_capacity);
  dptr->d_className = sidl_String_strdup(className);
  dptr->d_objectID = sidl_String_strdup(objectid);
  sidlx_rmi_Simvocation__set_data(self, dptr);

  /* Initialize Header */
  serialize(self,"EXEC:", 5, _ex);
  serialize(self,"objid:", 6, _ex);
  serialize(self,objectid, o_len, _ex);
  serialize(self,":clsid:", 7, _ex);
  serialize(self,className, c_len, _ex);
  serialize(self,":method:", 8, _ex);
  serialize(self,methodName, m_len, _ex);
  serialize(self,":args:", 6, _ex);
  return;
 EXIT:
  /*Not really anything to clean up...*/
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.init) */
#line 212 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  getMethodName[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_getMethodName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_Simvocation_getMethodName(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 220 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.getMethodName) */
  struct sidlx_rmi_Simvocation__data *dptr =
    sidlx_rmi_Simvocation__get_data(self);
  if(!dptr) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "This invocation has not been initialized!") 
  }
  return sidl_String_strdup(dptr->d_methodName);
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.getMethodName) */
#line 241 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  packBool[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_packBool"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_packBool(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ sidl_bool value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 249 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.packBool) */
  if(value) {
    char x = 0xFF;
    serialize(self, &x, 1, _ex); 
  }else{
    char x = 0;
    serialize(self, &x, 1, _ex); 
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.packBool) */
#line 271 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  packChar[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_packChar"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_packChar(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ char value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 277 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.packChar) */
  serialize(self, &value, 1, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.packChar) */
#line 295 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  packInt[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_packInt"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_packInt(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 299 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.packInt) */
  int32_t x = htonl(value);
  serialize(self, (char*)&x, 4, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.packInt) */
#line 320 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  packLong[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_packLong"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_packLong(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ int64_t value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 322 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.packLong) */
  short host = 1;
  short net = htons(host);
  if(host == net) {  /*This computer uses network byte ordering*/
    serialize(self, (char*)&value, 8, _ex); 
  } else {           /*This computer does not use network byte ordering*/
    flip64(&value);
    serialize(self, (char*)&value, 8, _ex); 
  }

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.packLong) */
#line 352 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  packFloat[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_packFloat"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_packFloat(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ float value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 352 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.packFloat) */
  /*TODO: Find out if this will work transfering to other platforms*/
  short host = 1;
  short net = htons(host);
  if(host == net) {  /*This computer uses network byte ordering*/
    serialize(self, (char*)&value, 4, _ex); 
  } else {           /*This computer does not use network byte ordering*/
    flip32((int32_t*)&value);
    serialize(self, (char*)&value, 4, _ex); 
  }

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.packFloat) */
#line 385 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  packDouble[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_packDouble"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_packDouble(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 383 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.packDouble) */
  /*TODO: Find out if this will work transfering to other platforms*/
  short host = 1;
  short net = htons(host);
  if(host == net) {  /*This computer uses network byte ordering*/
    serialize(self, (char*)&value, 8, _ex); 
  } else {           /*This computer does not use network byte ordering*/
    flip64((int64_t*)&value);
    serialize(self, (char*)&value, 8, _ex); 
  }

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.packDouble) */
#line 418 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  packFcomplex[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_packFcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_packFcomplex(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ struct sidl_fcomplex value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 414 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.packFcomplex) */
  /*TODO: Find out if this will work transfering to other platforms*/
  short host = 1;
  short net = htons(host);
  if(host == net) {  /*This computer uses network byte ordering*/
    serialize(self, (char*)&(value.real), 4, _ex); 
    serialize(self, (char*)&(value.imaginary), 4, _ex); 
  } else {           /*This computer does not use network byte ordering*/
    flip32((int32_t*)&(value.real));
    flip32((int32_t*)&(value.imaginary));
    serialize(self, (char*)&(value.real), 4, _ex); 
    serialize(self, (char*)&(value.imaginary), 4, _ex); 
  }

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.packFcomplex) */
#line 454 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  packDcomplex[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_packDcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_packDcomplex(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ struct sidl_dcomplex value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 448 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.packDcomplex) */
  /*TODO: Find out if this will work transfering to other platforms*/
  short host = 1;
  short net = htons(host);
  if(host == net) {  /*This computer uses network byte ordering*/
    serialize(self, (char*)&(value.real), 8, _ex); 
    serialize(self, (char*)&(value.imaginary), 8, _ex); 
  } else {           /*This computer does not use network byte ordering*/
    flip64((int64_t*)&(value.real));
    flip64((int64_t*)&(value.imaginary));
    serialize(self, (char*)&(value.real), 8, _ex); 
    serialize(self, (char*)&(value.imaginary), 8, _ex); 
  }


  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.packDcomplex) */
#line 491 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * Method:  packString[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_packString"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Simvocation_packString(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 483 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.packString) */
  int32_t len = sidl_String_strlen(value);
  int32_t f_len = htonl(len);
  serialize(self, (char*)&f_len, 4, _ex);
  serialize(self, value, len, _ex);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.packString) */
#line 518 "sidlx_rmi_Simvocation_Impl.c"
}

/*
 * this method may be called only once at the end of the object's lifetime 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Simvocation_invokeMethod"

#ifdef __cplusplus
extern "C"
#endif
sidl_rmi_Response
impl_sidlx_rmi_Simvocation_invokeMethod(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 506 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation.invokeMethod) */

  struct sidlx_rmi_Simvocation__data *dptr =
    sidlx_rmi_Simvocation__get_data(self);
  if(dptr) {
    sidlx_rmi_Simsponse sponse = NULL;
    sidl_rmi_Response ret = NULL;
    int lower = 0;
    int upper = (dptr->d_len)-1;
    int stride = 1;
    struct sidl_char__array *carray = sidl_char__array_borrow(dptr->d_buf,
							      1,&lower, &upper,&stride);
    
    sidlx_rmi_Socket_writestring(dptr->d_sock, dptr->d_len, carray, _ex);
    
    sponse = sidlx_rmi_Simsponse__create();
    sidlx_rmi_Simsponse_init(sponse, dptr->d_methodName, dptr->d_className, dptr->d_objectID,
			     dptr->d_sock, _ex); SIDL_CHECK(*_ex);

    ret = sidl_rmi_Response__cast(sponse);
    return ret;
  }
  SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simvocation has not been initialized");
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation.invokeMethod) */
#line 563 "sidlx_rmi_Simvocation_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_Response__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj) {
  return sidl_rmi_Response__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_Invocation__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj) {
  return sidl_rmi_Invocation__getURL(obj);
}
struct sidlx_rmi_Simvocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Simvocation__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Simvocation(struct 
  sidlx_rmi_Simvocation__object* obj) {
  return sidlx_rmi_Simvocation__getURL(obj);
}
struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_io_IOException__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) {
  return sidl_io_IOException__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_io_Serializer__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj) {
  return sidl_io_Serializer__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
