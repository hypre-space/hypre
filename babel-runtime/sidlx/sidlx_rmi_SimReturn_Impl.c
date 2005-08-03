/*
 * File:          sidlx_rmi_SimReturn_Impl.c
 * Symbol:        sidlx.rmi.SimReturn-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for sidlx.rmi.SimReturn
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.SimReturn" (version 0.1)
 * 
 * This type is used to pack return (out/inout) arguments on the server
 * side after a method has been exec'd.
 */

#include "sidlx_rmi_SimReturn_Impl.h"

#line 27 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._includes) */

#include <stdlib.h>
#include "sidl_String.h"
#include "sidl_Exception.h"

#include "sidlx_rmi_GenNetworkException.h"
/* Copies n elements from data into the vector we maintain as part of this 
   SimReturn object.  This function will realloc as nessecary */
static void serialize(sidlx_rmi_SimReturn self, const char* data, int n, sidl_BaseInterface* _ex) {
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
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
  sidlx_rmi_SimReturn__set_data(self, dptr);
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
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._includes) */
#line 72 "sidlx_rmi_SimReturn_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn__load(
  void)
{
#line 86 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._load) */
  /* insert implementation here: sidlx.rmi.SimReturn._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._load) */
#line 92 "sidlx_rmi_SimReturn_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn__ctor(
  /* in */ sidlx_rmi_SimReturn self)
{
#line 104 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._ctor) */
  /* insert implementation here: sidlx.rmi.SimReturn._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._ctor) */
#line 112 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn__dtor(
  /* in */ sidlx_rmi_SimReturn self)
{
#line 123 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._dtor) */
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  if(dptr) {
    free((void*)dptr->d_buf);
    sidlx_rmi_Socket_deleteRef(dptr->d_sock);
    sidl_String_free(dptr->d_methodName);
    free((void*)dptr);
    sidlx_rmi_SimReturn__set_data(self, NULL);
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._dtor) */
#line 141 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  init[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_init"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_init(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* methodName,
  /* in */ const char* className,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex)
{
#line 155 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.init) */
    struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  
  int m_len = sidl_String_strlen(methodName);
  int c_len = sidl_String_strlen(className);
  int o_len = sidl_String_strlen(objectid);
  int h_len = 5+6+o_len+7+c_len+8+m_len+6;  /*header length*/
  int t_capacity = h_len+128;
  /* Make this inital space for the function call equal the length of
   * EXEC:objid:<objectid>:clsid:<className>:method:<methodName>:args:(args go here)
   * Make the space for args be equal to 128 bytes (just a guess)
   */

  if (dptr) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "This inReturn has already been init'ed!");
  } else {
    dptr = malloc(sizeof(struct sidlx_rmi_SimReturn__data));
  }
  dptr->d_methodName = sidl_String_strdup(methodName);
  sidlx_rmi_Socket_addRef(sock);
  dptr->d_sock = sock;
  dptr->d_len = 0;
  dptr->d_capacity = t_capacity;
  dptr->d_buf = (char*)malloc(t_capacity);
  sidlx_rmi_SimReturn__set_data(self, dptr);

  /* Initialize Header */
  serialize(self,"RESP:", 5, _ex);
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
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.init) */
#line 205 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  getMethodName[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_getMethodName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimReturn_getMethodName(
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 213 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.getMethodName) */
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  if(!dptr) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "This inReturn has not been initialized!") 
  }
  return sidl_String_strdup(dptr->d_methodName);
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.getMethodName) */
#line 234 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * this method may be called only once at the end of the object's lifetime 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_SendReturn"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_SendReturn(
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 240 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.SendReturn) */
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  int lower = 0;
  int upper = (dptr->d_len)-1;
  int stride = 1;
  struct sidl_char__array *carray = sidl_char__array_borrow(dptr->d_buf,
							    1,&lower, &upper,&stride);
  
  
  sidlx_rmi_Socket_writestring(dptr->d_sock, dptr->d_len, carray, _ex);
  
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.SendReturn) */
#line 266 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  packBool[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packBool"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packBool(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ sidl_bool value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 272 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packBool) */
  if(value) {
    char x = 0xFF;
    serialize(self, &x, 1, _ex); 
  }else{
    char x = 0;
    serialize(self, &x, 1, _ex); 
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packBool) */
#line 296 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  packChar[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packChar"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packChar(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ char value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 300 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packChar) */
  serialize(self, &value, 1, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packChar) */
#line 320 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  packInt[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packInt"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packInt(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 322 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packInt) */
  int32_t x = htonl(value);
  serialize(self, (char*)&x, 4, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packInt) */
#line 345 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  packLong[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packLong"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packLong(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ int64_t value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 345 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packLong) */
  short host = 1;
  short net = htons(host);
  if(host == net) {  /*This computer uses network byte ordering*/
    serialize(self, (char*)&value, 8, _ex); 
  } else {           /*This computer does not use network byte ordering*/
    flip64(&value);
    serialize(self, (char*)&value, 8, _ex); 
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packLong) */
#line 376 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  packFloat[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packFloat"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packFloat(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ float value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 374 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packFloat) */
  short host = 1;
  short net = htons(host);
  if(host == net) {  /*This computer uses network byte ordering*/
    serialize(self, (char*)&value, 4, _ex); 
  } else {           /*This computer does not use network byte ordering*/
    flip32((int32_t*)&value);
    serialize(self, (char*)&value, 4, _ex); 
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packFloat) */
#line 407 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  packDouble[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packDouble"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packDouble(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 403 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packDouble) */
  short host = 1;
  short net = htons(host);
  if(host == net) {  /*This computer uses network byte ordering*/
    serialize(self, (char*)&value, 8, _ex); 
  } else {           /*This computer does not use network byte ordering*/
    flip64((int64_t*)&value);
    serialize(self, (char*)&value, 8, _ex); 
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packDouble) */
#line 438 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  packFcomplex[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packFcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packFcomplex(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ struct sidl_fcomplex value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 432 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packFcomplex) */
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
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packFcomplex) */
#line 472 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  packDcomplex[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packDcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packDcomplex(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ struct sidl_dcomplex value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 464 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packDcomplex) */
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
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packDcomplex) */
#line 506 "sidlx_rmi_SimReturn_Impl.c"
}

/*
 * Method:  packString[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packString"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packString(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
#line 496 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packString) */
  int32_t len = sidl_String_strlen(value);
  int32_t f_len = htonl(len);
  serialize(self, (char*)&f_len, 4, _ex);
  serialize(self, value, len, _ex);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packString) */
#line 533 "sidlx_rmi_SimReturn_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_SimReturn_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimReturn__connect(url, _ex);
}
char * impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_SimReturn(struct 
  sidlx_rmi_SimReturn__object* obj) {
  return sidlx_rmi_SimReturn__getURL(obj);
}
struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_io_IOException__connect(url, _ex);
}
char * impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) {
  return sidl_io_IOException__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_SimReturn_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_io_Serializer__connect(url, _ex);
}
char * impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj) {
  return sidl_io_Serializer__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
