/*
 * File:          sidlx_rmi_SimReturn_Impl.c
 * Symbol:        sidlx.rmi.SimReturn-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for sidlx.rmi.SimReturn
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.SimReturn" (version 0.1)
 * 
 * This type is used to pack return (out/inout) arguments on the server
 * side after a method has been exececuted.
 */

#include "sidlx_rmi_SimReturn_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._includes) */

#include <stdlib.h>
#include "sidl_String.h"
#include "sidl_Exception.h"
#include "sidlx_common.h"
#include <string.h>
#include "sidl_rmi_NetworkException.h"
#include "sidl_io_Serializable.h"
#include "sidl_CastException.h"
/* Serialize the response into a buffer and send!*/

/* Copies n elements from data into the vector we maintain as part of this 
   SimReturn object.  This function will realloc as nessecary */
static void serialize(sidlx_rmi_SimReturn self, const char* data, int n, sidl_BaseInterface* _ex) {
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  if(dptr) {
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
    return;
  } 
  SIDL_THROW(*_ex, sidl_rmi_NetworkException, "This Return has not been init'ed!");
 EXIT:
  return;
}
/* Allocates a bunch of space on the buffer for an incoming array.  May or may not 
   actually need to do memory allocation.  Returns a pointer to the beginning of the
   space where the array should go.*/
static void* buffer_alloc(sidlx_rmi_SimReturn self, int64_t total_len, sidl_BaseInterface* _ex ) {
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  if(dptr) {
    int rem = dptr->d_capacity - dptr->d_len; /*space remaining*/
    char* s_ptr = NULL; 
    
    if(rem < total_len) {
      dptr->d_capacity += total_len;  /*Give us just enough space (arrays might be big)*/
      dptr->d_buf = (char*)realloc((void*)dptr->d_buf, dptr->d_capacity);
    }
    s_ptr = (dptr->d_buf)+(dptr->d_len); /* get pointer to beginning of space*/
    sidlx_rmi_SimReturn__set_data(self, dptr);/*TODO: Eliminate this?*/
    (dptr->d_len) += total_len; 
    return s_ptr;
  }
  SIDL_THROW(*_ex, sidl_rmi_NetworkException, "This Return has not been init'ed!");
 EXIT:
  return NULL;
}

/**
 * Resets the serialization buffer to the beginning.  Useful for exceptions
 */
static void buffer_reset(sidlx_rmi_SimReturn self, sidl_BaseInterface* _ex ) {
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  if(dptr) {
    dptr->d_len = dptr->d_begin;
    return;
  }
  SIDL_THROW(*_ex, sidl_rmi_NetworkException, "This Return has not been init'ed!");
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

static void* prep_array(sidlx_rmi_SimReturn self, struct sidl__array* value, 
			int32_t ordering, int32_t dimen, sidl_bool reuse,
			int32_t obj_size, int32_t obj_per_elem, int32_t* dest_stride,
			int32_t* lengths, int32_t* current, sidl_BaseInterface* _ex) {
  sidl_bool isRow = FALSE;
  char* srcFirst = NULL; 
  void* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t *src_stride = NULL; 
  int i;
  int real_ordering = 0;
  
  /* If the array is null, or a dimension is required but it different than
     the passed in array, pass a null array*/
  if(!value || ((dimen != 0) && value->d_dimen != dimen)) {
    /*serilize isRow*/
    impl_sidlx_rmi_SimReturn_packBool(self, NULL, FALSE, _ex); SIDL_CHECK(*_ex);
    /*serialize dimension*/
    impl_sidlx_rmi_SimReturn_packInt(self, NULL, 0, _ex); SIDL_CHECK(*_ex);
    return NULL;
  }

  isRow =  sidl__array_isRowOrder(value);
  srcFirst = sidl_char__array_first((struct sidl_char__array*)value);
  l_dimen =  value->d_dimen;
  src_stride = value->d_stride;

  if(ordering) {
    real_ordering = ordering;
  } else {
    if(sidl__array_isRowOrder(value)) {
      real_ordering = sidl_row_major_order;
    }else {
      real_ordering = sidl_column_major_order;
    }
  }

  if(real_ordering == sidl_row_major_order) {
    int32_t size=1;
    for(i = l_dimen-1; i >= 0; --i) {
      dest_stride[i] = size;
      size *= (1 + (value->d_upper)[i] - (value->d_lower)[i]);
    }
    isRow = TRUE;
  } else {
    int32_t size=1;
    for(i = 0; i < l_dimen; ++i) {
      dest_stride[i] = size;
      size *= (1 + (value->d_upper)[i] - (value->d_lower)[i]);
    }
    isRow = FALSE;
  }


  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(value, count);
    t_len *= len;
    lengths[count] = len;
    current[count] = 0;
  }

  /*serilize isRow*/
  impl_sidlx_rmi_SimReturn_packBool(self, NULL, isRow, _ex); SIDL_CHECK(*_ex);
  /*serialize dimension*/
  impl_sidlx_rmi_SimReturn_packInt(self, NULL, l_dimen, _ex); SIDL_CHECK(*_ex);
  /*Serialize resue boolean*/
  impl_sidlx_rmi_SimReturn_packBool(self, NULL, reuse, _ex); SIDL_CHECK(*_ex);
  /*serialize upper lower stride * dim */
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_SimReturn_packInt(self, NULL, (value->d_lower)[count], _ex); SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_SimReturn_packInt(self, NULL, (value->d_upper)[count], _ex); SIDL_CHECK(*_ex);
  }

  /*serialize data */
  /*Allocate enough buffer space for the array, and put the pointer in destFirst*/ 
  destFirst = buffer_alloc(self, t_len*obj_size*obj_per_elem, _ex);  SIDL_CHECK(*_ex);
  return destFirst;
 EXIT:
  return NULL;
}

/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._load) */
  /* insert implementation here: sidlx.rmi.SimReturn._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._load) */
  }
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
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._ctor) */
  /* insert implementation here: sidlx.rmi.SimReturn._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn__ctor2(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.SimReturn._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._ctor2) */
  }
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
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._dtor) */
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  if(dptr) {
    free((void*)dptr->d_buf);
    sidlx_rmi_Socket_deleteRef(dptr->d_sock, _ex); SIDL_CHECK(*_ex);
    sidl_String_free(dptr->d_methodName);
    free((void*)dptr);
    sidlx_rmi_SimReturn__set_data(self, NULL);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._dtor) */
  }
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
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.init) */
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  
  int m_len = sidl_String_strlen(methodName);
  /*  int c_len = sidl_String_strlen(className);*/
  int o_len = sidl_String_strlen(objectid);
  int h_len = 5+6+o_len+7+8+m_len+6;  /*header length (used to have c_len)*/
  int t_capacity = h_len+128;

  /* Make this inital space for the function call equal the length of
   * EXEC:objid:<objectid>:clsid:<className>:method:<methodName>:args:(args go here)
   * Make the space for args be equal to 128 bytes (just a guess)
   * Note: The first 'arg' is always a boolean that is true if an exception is thrown.
   */

  if (dptr) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "This inReturn has already been init'ed!");
  } else {
    dptr = malloc(sizeof(struct sidlx_rmi_SimReturn__data));
  }
  dptr->d_methodName = sidl_String_strdup(methodName);
  sidlx_rmi_Socket_addRef(sock,_ex); SIDL_CHECK(*_ex);
  dptr->d_sock = sock;
  dptr->d_len = 0;
  dptr->d_capacity = t_capacity;
  dptr->d_buf = (char*)malloc(t_capacity);
  sidlx_rmi_SimReturn__set_data(self, dptr);

  /* Initialize Header */
  serialize(self,"RESP:", 5, _ex);
  serialize(self,"objid:", 6, _ex);
  serialize(self,objectid, o_len, _ex);
  /*  serialize(self,":clsid:", 7, _ex);
      serialize(self,className, c_len, _ex);*/
  serialize(self,":method:", 8, _ex);
  serialize(self,methodName, m_len, _ex);
  serialize(self,":args:", 6, _ex);

  dptr->d_begin = dptr->d_len; /* Set where the args begin*/ 
  /* No exception thrown yet*/
  sidlx_rmi_SimReturn_packBool(self, "ex_thrown", 0, _ex); SIDL_CHECK(*_ex);
  return;
 EXIT:
  /*Not really anything to clean up...*/
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.init) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.getMethodName) */
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  if(!dptr) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "This Return has not been init'd!") 
      }
  return sidl_String_strdup(dptr->d_methodName);
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.getMethodName) */
  }
}

/*
 *  this method is the last call on a return object.
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.SendReturn) */
  struct sidlx_rmi_SimReturn__data *dptr =
    sidlx_rmi_SimReturn__get_data(self);
  if(dptr) {
    int lower = 0;
    int upper = (dptr->d_len)-1;
    int stride = 1;
    struct sidl_char__array *carray = 
      sidl_char__array_borrow(dptr->d_buf,1,&lower, &upper,&stride);
    
    sidlx_rmi_Socket_writestring(dptr->d_sock, dptr->d_len, carray, _ex);
    
    sidl_char__array_deleteRef(carray);
    return;
  }
  SIDL_THROW(*_ex, sidl_rmi_NetworkException, "This Return has not been init'ed!");
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.SendReturn) */
  }
}

/*
 *  
 * This method serialized exceptions thrown on the server side
 * that should be returned to the client.  Assumed to invalidate
 * in previously serialized arguments.  (Also assumed that no
 * more arguments will be serialized.)
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_throwException"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_throwException(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ sidl_BaseException ex_to_throw,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.throwException) */
  sidl_BaseInterface _throwaway_exception = NULL;
  sidl_io_Serializable ex_tmp = 
    sidl_io_Serializable__cast(ex_to_throw,_ex); SIDL_CHECK(*_ex);
  buffer_reset(self, _ex); SIDL_CHECK(*_ex);
  sidlx_rmi_SimReturn_packBool(self, "ex_thrown", 1, _ex); SIDL_CHECK(*_ex);
  sidlx_rmi_SimReturn_packSerializable(self, "_ex", ex_tmp, _ex); SIDL_CHECK(*_ex);
 EXIT:
  if (ex_tmp) sidl_io_Serializable_deleteRef(ex_tmp,&_throwaway_exception);
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.throwException) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packBool) */
  if(value) {
    char x = 0xFF;
    serialize(self, &x, 1, _ex); 
  }else{
    char x = 0;
    serialize(self, &x, 1, _ex); 
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packBool) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packChar) */
  serialize(self, &value, 1, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packChar) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packInt) */
  int32_t x = htonl(value);
  serialize(self, (char*)&x, 4, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packInt) */
  }
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
  *_ex = 0;
  {
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
  }
}

/*
 * Method:  packOpaque[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packOpaque"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packOpaque(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ void* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packOpaque) */
  short host = 1;
  short net = htons(host);
  int64_t temp = (int64_t) (ptrdiff_t) value;
  if(host == net) {  /*This computer uses network byte ordering*/
    serialize(self, (char*)&temp, 8, _ex); 
  } else {           /*This computer does not use network byte ordering*/
    flip64(&temp);
    serialize(self, (char*)&temp, 8, _ex); 
  }

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packOpaque) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packFloat) */
  serialize(self, (char*)&value, 4, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packFloat) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packDouble) */
  serialize(self, (char*)&value, 8, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packDouble) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packFcomplex) */
  serialize(self, (char*)&(value.real), 4, _ex); 
  serialize(self, (char*)&(value.imaginary), 4, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packFcomplex) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packDcomplex) */
  serialize(self, (char*)&(value.real), 8, _ex); 
  serialize(self, (char*)&(value.imaginary), 8, _ex); 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packDcomplex) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packString) */
  int32_t len = sidl_String_strlen(value);
  int32_t f_len = htonl(len);
  serialize(self, (char*)&f_len, 4, _ex);
  serialize(self, value, len, _ex);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packString) */
  }
}

/*
 * Method:  packSerializable[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packSerializable"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packSerializable(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ sidl_io_Serializable value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packSerializable) */
  sidl_BaseInterface _throwaway_exception = NULL;
  sidl_io_Serializer serial = NULL;
  sidl_ClassInfo ci = NULL;
  char* class_name = NULL;
  char* obj_url = NULL;
  int is_remote = sidl_io_Serializable__isRemote(value, _ex); SIDL_CHECK(*_ex);
  if(is_remote) {
    sidlx_rmi_SimReturn_packBool(self, NULL, is_remote, _ex); SIDL_CHECK(*_ex);
    obj_url = sidl_io_Serializable__getURL(value, _ex);SIDL_CHECK(*_ex);
    sidlx_rmi_SimReturn_packString(self, NULL, obj_url, _ex); SIDL_CHECK(*_ex);
    
  } else {
    sidlx_rmi_SimReturn_packBool(self, NULL, is_remote, _ex); SIDL_CHECK(*_ex);
    ci = sidl_io_Serializable_getClassInfo(value,_ex); SIDL_CHECK(*_ex);
    class_name = sidl_ClassInfo_getName(ci,_ex); SIDL_CHECK(*_ex);
    sidlx_rmi_SimReturn_packString(self, NULL, class_name, _ex); SIDL_CHECK(*_ex);
    serial = sidl_io_Serializer__cast(self,_ex); SIDL_CHECK(*_ex);
    sidl_io_Serializable_packObj(value, serial, _ex); SIDL_CHECK(*_ex); 
  }
 EXIT:
  if(ci) {sidl_ClassInfo_deleteRef(ci,&_throwaway_exception);}
  if(serial) {sidl_io_Serializer_deleteRef(serial, &_throwaway_exception);}
  sidl_String_free(class_name);
  sidl_String_free(obj_url);
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packSerializable) */
  }
}

/*
 *  
 * pack arrays of values.  It is possible to ensure an array is
 * in a certain order by passing in ordering and dimension
 * requirements.  ordering should represent a value in the
 * sidl_array_ordering enumeration in sidlArray.h If either
 * argument is 0, it means there is no restriction on that
 * aspect.  The boolean reuse_array flag is set to true if the
 * remote unserializer should try to reuse the array that is
 * passed into it or not.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packBoolArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packBoolArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<bool> */ struct sidl_bool__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packBoolArray) */
  
  sidl_bool* srcFirst = NULL; 
  char* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  destFirst = (char*)prep_array(self, (struct sidl__array*)value, ordering, 
				dimen, reuse_array, 1,1,dest_stride, 
				lengths, current, _ex);
  SIDL_CHECK(*_ex);

  if(destFirst) {
    src_stride = value->d_metadata.d_stride;
    srcFirst = sidl_bool__array_first(value);
    l_dimen = sidlArrayDim(value);
    for(i=0; i<l_dimen; ++i) {
      t_len *= lengths[i];
    }
    
    if(t_len > 0) {
      do {
	if(*srcFirst) {
	  *destFirst = (char)0xFF;
	} else {
	  *destFirst = (char)0;
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
    /* Finished serializing data*/
  }
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packBoolArray) */
  }
}

/*
 * Method:  packCharArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packCharArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packCharArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<char> */ struct sidl_char__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packCharArray) */
 
  char* srcFirst = NULL; 
  char* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  destFirst = (char*)prep_array(self, (struct sidl__array*)value, ordering, 
				dimen, reuse_array, 1,1,dest_stride, lengths, current, _ex);
  SIDL_CHECK(*_ex);

  if(destFirst) {
    src_stride = value->d_metadata.d_stride;
    srcFirst = sidl_char__array_first(value);
    l_dimen = sidlArrayDim(value);
    for(i=0; i<l_dimen; ++i) {
      t_len *= lengths[i];
    }
    
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
  }
  /* Finished serializing data*/

 EXIT:
  return;
 
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packCharArray) */
  }
}

/*
 * Method:  packIntArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packIntArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packIntArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<int> */ struct sidl_int__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packIntArray) */
    
  int32_t* srcFirst = NULL; 
  int32_t* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  destFirst = (int32_t*)prep_array(self, (struct sidl__array*)value, ordering, 
				   dimen, reuse_array, 4,1,dest_stride, lengths, current, _ex);
  SIDL_CHECK(*_ex);

  if(destFirst) {
    src_stride = value->d_metadata.d_stride;
    srcFirst = sidl_int__array_first(value);
    l_dimen = sidlArrayDim(value);
    for(i=0; i<l_dimen; ++i) {
      t_len *= lengths[i];
    }
    
    if(t_len > 0) {
      do {
	*destFirst = htonl(*srcFirst);
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
  }
  /* Finished serializing data*/

 EXIT:
  return;
  
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packIntArray) */
  }
}

/*
 * Method:  packLongArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packLongArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packLongArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<long> */ struct sidl_long__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packLongArray) */

  int64_t* srcFirst = NULL; 
  int64_t* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  destFirst = (int64_t*)prep_array(self, (struct sidl__array*)value, ordering, 
				   dimen, reuse_array,8,1,dest_stride, lengths, current, _ex);
  SIDL_CHECK(*_ex);

  if(destFirst) {
    short host = 1;
    short net = htons(host);
    src_stride = value->d_metadata.d_stride;
    srcFirst = sidl_long__array_first(value);
    l_dimen = sidlArrayDim(value);
       
    for(i=0; i<l_dimen; ++i) {
      t_len *= lengths[i];
    }
    
    if(t_len > 0) {
      do {
	if(host == net) {  /*This computer uses network byte ordering*/
	  *destFirst = *srcFirst;
	} else {           /*This computer does not use network byte ordering*/
	  int64_t tmp = *srcFirst;
	  flip64(&tmp);
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
  }
  /* Finished serializing data*/

 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packLongArray) */
  }
}

/*
 * Method:  packOpaqueArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packOpaqueArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packOpaqueArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<opaque> */ struct sidl_opaque__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packOpaqueArray) */

  void** srcFirst = NULL; 
  int64_t* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int64_t temp = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  destFirst = (int64_t*)prep_array(self, (struct sidl__array*)value, ordering, 
				   dimen, reuse_array, 
				   8,1,dest_stride, lengths, current, _ex);
  SIDL_CHECK(*_ex);

  if(destFirst) {
    short host = 1;
    short net = htons(host);
    src_stride = value->d_metadata.d_stride;
    srcFirst = (void**) sidl_long__array_first((struct sidl_long__array*)value);
    l_dimen = sidlArrayDim(value);
    for(i=0; i<l_dimen; ++i) {
      t_len *= lengths[i];
    }
    
    if(t_len > 0) {
      do {
	temp = (int64_t) (ptrdiff_t)*srcFirst;
	if(host == net) {  /*This computer uses network byte ordering*/
	  *destFirst = temp;
	} else {           /*This computer does not use network byte ordering*/
	  flip64(&temp);
	  *destFirst = temp;
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
  }
  /* Finished serializing data*/

 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packOpaqueArray) */
  }
}

/*
 * Method:  packFloatArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packFloatArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packFloatArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<float> */ struct sidl_float__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packFloatArray) */

  float* srcFirst = NULL; 
  float* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  destFirst = (float*)prep_array(self, (struct sidl__array*)value, ordering, 
				 dimen, reuse_array,4,1,dest_stride, lengths, current, _ex);
  SIDL_CHECK(*_ex);

  if(destFirst) {
    src_stride = value->d_metadata.d_stride;
    srcFirst = sidl_float__array_first(value);
    l_dimen = sidlArrayDim(value);
       
    for(i=0; i<l_dimen; ++i) {
      t_len *= lengths[i];
    }
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
  }
  /* Finished serializing data*/

 EXIT:
  return;
  
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packFloatArray) */
  }
}

/*
 * Method:  packDoubleArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packDoubleArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packDoubleArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<double> */ struct sidl_double__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packDoubleArray) */

  double* srcFirst = NULL; 
  double* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 

  destFirst = (double*)prep_array(self, (struct sidl__array*)value, ordering, 
				  dimen, reuse_array,8,1,dest_stride, lengths, current, _ex);
  SIDL_CHECK(*_ex);

  if(destFirst) {
    src_stride = value->d_metadata.d_stride;
    srcFirst = sidl_double__array_first(value);
    l_dimen = sidlArrayDim(value);
      
    for(i=0; i<l_dimen; ++i) {
      t_len *= lengths[i];
    }
    
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
  }
  /* Finished serializing data*/

 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packDoubleArray) */
  }
}

/*
 * Method:  packFcomplexArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packFcomplexArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packFcomplexArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<fcomplex> */ struct sidl_fcomplex__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packFcomplexArray) */

  struct sidl_fcomplex* srcFirst = NULL; 
  struct sidl_fcomplex* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 

  destFirst = (struct sidl_fcomplex*)prep_array(self, (struct sidl__array*)value, ordering, 
						dimen, reuse_array, 4,2,dest_stride, lengths, current, _ex);
  SIDL_CHECK(*_ex);

  if(destFirst) {
    src_stride = value->d_metadata.d_stride;
    srcFirst = sidl_fcomplex__array_first(value);
    l_dimen = sidlArrayDim(value);
       
    for(i=0; i<l_dimen; ++i) {
      t_len *= lengths[i];
    }
    
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
  }
  /* Finished serializing data*/

 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packFcomplexArray) */
  }
}

/*
 * Method:  packDcomplexArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packDcomplexArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packDcomplexArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<dcomplex> */ struct sidl_dcomplex__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packDcomplexArray) */

  struct sidl_dcomplex* srcFirst = NULL; 
  struct sidl_dcomplex* destFirst = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  destFirst = (struct sidl_dcomplex*)prep_array(self, (struct sidl__array*)value, ordering, 
						dimen, reuse_array, 8,2,dest_stride, lengths, current, _ex);
  SIDL_CHECK(*_ex);

  if(destFirst) {
    src_stride = value->d_metadata.d_stride;
    srcFirst = sidl_dcomplex__array_first(value);
    l_dimen = sidlArrayDim(value);
    for(i=0; i<l_dimen; ++i) {
      t_len *= lengths[i];
    }
    
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
  }
  /* Finished serializing data*/

 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packDcomplexArray) */
  }
}

/*
 * Method:  packStringArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packStringArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packStringArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<string> */ struct sidl_string__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packStringArray) */
  sidl_bool isRow = FALSE;
  char** srcFirst = NULL; 
  /*char* destFirst = NULL;*/
  int32_t l_dimen = 0; /*dimension (local copy)*/
  /*int64_t t_len = 1; Total length (of the array, in elements)*/ 
  int32_t count = 0;
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int real_ordering = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  /* If the array is null, or a dimension is required but it different than
     the passed in array, pass a null array*/
  if(!value || ((dimen != 0) && sidl_string__array_dimen(value) != dimen)) {
    /*serilize isRow*/
    impl_sidlx_rmi_SimReturn_packBool(self, NULL, FALSE, _ex); SIDL_CHECK(*_ex);
    /*serialize dimension*/
    impl_sidlx_rmi_SimReturn_packInt(self, NULL, 0, _ex); SIDL_CHECK(*_ex);
    return;
  }

  isRow =  sidl_string__array_isRowOrder(value);
  /* HACK! */
  srcFirst = (char**)sidl_int__array_first((struct sidl_int__array*)value);
  l_dimen =  sidl_string__array_dimen(value);
  src_stride = ((struct sidl__array*)value)->d_stride;

  if(ordering) {
    real_ordering = ordering;
  } else {
    if(sidl_string__array_isRowOrder(value)) {
      real_ordering = sidl_row_major_order;
    }else {
      real_ordering = sidl_column_major_order;
    }
  }

  if(real_ordering == sidl_row_major_order) {
    int32_t size=1;
    for(i = l_dimen-1; i >= 0; --i) {
      dest_stride[i] = size;
      size *= sidl_string__array_length(value, i);
    }
    isRow = TRUE;
  } else {
    int32_t size=1;
    for(i = 0; i < l_dimen; ++i) {
      dest_stride[i] = size;
      size *=  sidl_string__array_length(value, i);
    }
    isRow = FALSE;
  }


  /*Figure out lengths (useful for copy)*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(value, count);
    lengths[count] = len;
    t_len *= len;
    current[count] = 0;
  }

  /*serilize isRow*/
  impl_sidlx_rmi_SimReturn_packBool(self, NULL, isRow, _ex); SIDL_CHECK(*_ex);
  /*serialize dimension*/
  impl_sidlx_rmi_SimReturn_packInt(self, NULL, l_dimen, _ex); SIDL_CHECK(*_ex);
  /*serialize reuse bool*/
  impl_sidlx_rmi_SimReturn_packBool(self, NULL, reuse_array, _ex); SIDL_CHECK(*_ex);

  /*serialize upper lower stride * dim */
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_SimReturn_packInt(self, NULL, sidl_string__array_lower(value, count), _ex); SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_SimReturn_packInt(self, NULL, sidl_string__array_upper(value, count), _ex); SIDL_CHECK(*_ex);
  }
  /* Serialization of stride is unessecary, it is known from above info*/

  /*serialize data */
  /*Allocate enough buffer space for the array, and put the pointer in destFirst*/ 
  if(t_len > 0) {
    do {
      int32_t len = sidl_String_strlen(*srcFirst);
      int32_t f_len = htonl(len);
      serialize(self, (char*)&f_len, 4, _ex);SIDL_CHECK(*_ex);
      serialize(self, *srcFirst, len, _ex);SIDL_CHECK(*_ex);
    
      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
  /* Finished serializing data*/

 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packStringArray) */
  }
}

/*
 * Method:  packGenericArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packGenericArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packGenericArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<> */ struct sidl__array* value,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packGenericArray) */
  int32_t type = 0;
  if(value){
    type = sidl__array_type(value);
  } else {
    /* If the array is null, pass a null array and bail */
    impl_sidlx_rmi_SimReturn_packInt(self, NULL, 0, _ex); SIDL_CHECK(*_ex);
    return;
  }
  
  /*serialize type */
  impl_sidlx_rmi_SimReturn_packInt(self, NULL, type, _ex); SIDL_CHECK(*_ex);
  switch(type) {
  case sidl_bool_array:
    sidlx_rmi_SimReturn_packBoolArray(self,key,(struct sidl_bool__array*) value, 
				      0,0,reuse_array,_ex);
    break;
  case sidl_char_array:
    sidlx_rmi_SimReturn_packCharArray(self,key,(struct sidl_char__array*) value, 
				      0,0,reuse_array,_ex);
    break;
  case sidl_dcomplex_array:
    sidlx_rmi_SimReturn_packDcomplexArray(self,key,(struct sidl_dcomplex__array*) value, 
					  0,0,reuse_array,_ex);
    break;
  case sidl_double_array:
    sidlx_rmi_SimReturn_packDoubleArray(self,key,(struct sidl_double__array*) value, 
					0,0,reuse_array,_ex);
    break;
  case sidl_fcomplex_array:
    sidlx_rmi_SimReturn_packFcomplexArray(self,key,(struct sidl_fcomplex__array*) value, 
					  0,0,reuse_array,_ex);
    break;
  case sidl_float_array:
    sidlx_rmi_SimReturn_packFloatArray(self,key,(struct sidl_float__array*) value, 
				       0,0,reuse_array,_ex);
    break;
  case sidl_int_array:
    sidlx_rmi_SimReturn_packIntArray(self,key,(struct sidl_int__array*) value, 
				     0,0,reuse_array,_ex);
    break;
  case sidl_long_array:
    sidlx_rmi_SimReturn_packLongArray(self,key,(struct sidl_long__array*) value, 
				      0,0,reuse_array,_ex);
    break;
  case sidl_opaque_array:
    sidlx_rmi_SimReturn_packOpaqueArray(self,key,(struct sidl_opaque__array*) value, 
					0,0,reuse_array,_ex);
    break;
  case sidl_string_array:
    sidlx_rmi_SimReturn_packStringArray(self,key,(struct sidl_string__array*) value, 
					0,0,reuse_array,_ex);
    break;
  case sidl_interface_array:
    
    sidlx_rmi_SimReturn_packSerializableArray(self,key,(struct sidl_io_Serializable__array*) value, 
					      0,0,reuse_array,_ex);
  }
 EXIT:  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packGenericArray) */
  }
}

/*
 * Method:  packSerializableArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimReturn_packSerializableArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimReturn_packSerializableArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<sidl.io.Serializable> */ struct sidl_io_Serializable__array* 
    value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn.packSerializableArray) */
  sidl_bool isRow = FALSE;
  sidl_io_Serializable* srcFirst = NULL; 
  sidl_io_Serializable toSerialize = NULL;
  sidl_BaseInterface toCast = NULL;
  int32_t l_dimen = 0; /*dimension (local copy)*/
  int32_t count = 0;
  int32_t dest_stride[7];
  int32_t *src_stride = NULL; 
  int32_t lengths[7];
  int32_t current[7];
  int i;
  int real_ordering = 0;
  int64_t t_len = 1; /*Total length (of the array, in elements)*/ 
  /* If the array is null, or a dimension is required but it different than
     the passed in array, pass a null array*/
  if(!value || ((dimen != 0) && sidl_io_Serializable__array_dimen(value) != dimen)) {
    /*serilize isRow*/
    impl_sidlx_rmi_SimReturn_packBool(self, NULL, FALSE, _ex); SIDL_CHECK(*_ex);
    /*serialize dimension*/
    impl_sidlx_rmi_SimReturn_packInt(self, NULL, 0, _ex); SIDL_CHECK(*_ex);
    return;
  }

  isRow =  sidl_io_Serializable__array_isRowOrder(value);
  l_dimen =  sidl_io_Serializable__array_dimen(value);
  /*HACK*/
  srcFirst = (sidl_io_Serializable*)sidl_char__array_first((struct sidl_char__array*)value);
  src_stride = ((struct sidl_char__array*)value)->d_metadata.d_stride;

  if(ordering) {
    real_ordering = ordering;
  } else {
    if(sidl_io_Serializable__array_isRowOrder(value)) {
      real_ordering = sidl_row_major_order;
    }else {
      real_ordering = sidl_column_major_order;
    }
  }

  if(real_ordering == sidl_row_major_order) {
    int32_t size=1;
    for(i = l_dimen-1; i >= 0; --i) {
      dest_stride[i] = size;
      size *= sidl_io_Serializable__array_length(value, i);
    }
    isRow = TRUE;
  } else {
    int32_t size=1;
    for(i = 0; i < l_dimen; ++i) {
      dest_stride[i] = size;
      size *= sidl_io_Serializable__array_length(value, i);
    }
    isRow = FALSE;
  }


  /*Figure out lengths (useful for copy)*/
  for(count=0; count<l_dimen; ++count) {
    int32_t len = sidlLength(value, count);
    lengths[count] = len;
    t_len *= len;
    current[count] = 0;
  }

  /*serilize isRow*/
  impl_sidlx_rmi_SimReturn_packBool(self, NULL, isRow, _ex); SIDL_CHECK(*_ex);
  /*serialize dimension*/
  impl_sidlx_rmi_SimReturn_packInt(self, NULL, l_dimen, _ex); SIDL_CHECK(*_ex);
  /*serialize reuse bool*/
  impl_sidlx_rmi_SimReturn_packBool(self, NULL, reuse_array, _ex); SIDL_CHECK(*_ex);

  /*serialize upper lower stride * dim */
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_SimReturn_packInt(self, NULL,  sidl_io_Serializable__array_lower(value,count), _ex); SIDL_CHECK(*_ex);
  }
  for(count = 0; count < l_dimen; ++count) {
    impl_sidlx_rmi_SimReturn_packInt(self, NULL,  sidl_io_Serializable__array_upper(value,count), _ex); SIDL_CHECK(*_ex);
  }
  /* Serialization of stride is unessecary, it is known from above info*/

  /*serialize data */
  /*Allocate enough buffer space for the array, and put the pointer in destFirst*/ 
 
  if(t_len > 0) {
    do {
      toCast = (sidl_BaseInterface) *srcFirst;
      toSerialize = sidl_io_Serializable__cast(toCast,_ex); SIDL_CHECK(*_ex);
      if(toSerialize == NULL) {
	SIDL_THROW(*_ex, sidl_CastException, "Attempted to serialize a non-serializable object in an object array");
      }
      sidlx_rmi_SimReturn_packSerializable(self, NULL, toSerialize, _ex); SIDL_CHECK(*_ex);
      sidl_io_Serializable_deleteRef(toSerialize, _ex); SIDL_CHECK(*_ex);
      /* the whole point of this for-loop is to move forward one element */
      for(i = l_dimen - 1; i >= 0; --i) {
	++(current[i]);
	if (current[i] >= lengths[i]) {
	  /* this dimension has been enumerated already reset to beginning */
	  current[i] = 0;
	  /* prepare to next iteration of for-loop for i-1 */
	  srcFirst -= ((lengths[i]-1) * src_stride[i]);
	}
	else {
	  /* move forward one element in dimension i */
	  srcFirst += src_stride[i];
	  break; /* exit for loop */
	}
      }
    } while (i >= 0);
  }
  /* Finished serializing data*/

 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn.packSerializableArray) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_sidlx_rmi_SimReturn_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseException__connectI(url, ar, _ex);
}
struct sidl_BaseException__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_BaseException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseException__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_sidlx_rmi_SimReturn_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializable(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializable__connectI(url, ar, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializable__cast(bi, _ex);
}
struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializer__connectI(url, ar, _ex);
}
struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_io_Serializer(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializer__cast(bi, _ex);
}
struct sidl_rmi_Return__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_Return(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_Return__connectI(url, ar, _ex);
}
struct sidl_rmi_Return__object* impl_sidlx_rmi_SimReturn_fcast_sidl_rmi_Return(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_rmi_Return__cast(bi, _ex);
}
struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimReturn__connectI(url, ar, _ex);
}
struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidlx_rmi_SimReturn(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_SimReturn__cast(bi, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidlx_rmi_Socket(void* bi, sidl_BaseInterface* 
  _ex) {
  return sidlx_rmi_Socket__cast(bi, _ex);
}
