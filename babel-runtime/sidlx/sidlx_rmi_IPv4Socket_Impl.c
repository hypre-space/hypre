/*
 * File:          sidlx_rmi_IPv4Socket_Impl.c
 * Symbol:        sidlx.rmi.IPv4Socket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for sidlx.rmi.IPv4Socket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.10
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.IPv4Socket" (version 0.1)
 * 
 *  Basic functionality for an IPv4 Socket.  Implements most of the functions in Socket
 */

#include "sidlx_rmi_IPv4Socket_Impl.h"

#line 26 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket._includes) */
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "sidlx_rmi_GenNetworkException.h"
#include "sidl_String.h"
#include "sidl_Exception.h"
/* this is a utility function that makes sure a char array is
   1-D, packed, and has a minimum length. */
void ensure1DPackedChar( const int32_t minlen, 
			      struct sidl_char__array ** data ) { 
  int realloc = 1;
  int len;

  if (*data != NULL) { 
    if ( (sidl_char__array_dimen(*data)==1) &&  /* if 1-D */
	 (sidl_char__array_stride(*data,0)==1) ) {  /* and packed */
      len = sidl_char__array_length(*data,0); /* get length */
      if ( len >= minlen ) {                  /* if long enough */
	realloc = 0; /* no realloc */
      }
    }
    if (realloc) {  /* if realloc, then free current array */
      sidl_char__array_deleteRef(*data);
      *data=NULL;
    }
  }
  if (*data==NULL) { 
    /* at this point, whether it was always NULL or recently realloced 
       doesn't matter */
    *data = sidl_char__array_create1d(minlen+1);
  }
  return;
}

/* read nbytes into a character string */
int32_t readn2( int filedes, int32_t nbytes, char ** data,
		 sidl_BaseInterface *_ex) {
  size_t nleft;
  ssize_t nread;
  char* ptr = *data;

  if ( *data == NULL ) { 
    *data = sidl_String_alloc(nbytes);
  }

  nleft = nbytes;
  
  while ( nleft > 0 ) {
    if ( ( nread = read( filedes, ptr, nleft)) < 0 ) { 
      if ( errno == EINTR ) { 
	nread = 0; /* and call read() again */
      } else { 
	nleft = nbytes+1;
	SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "readn() error!");
      } 
    } else if ( nread == 0 ) { 
      break; /* EOF */
    }
    nleft -= nread;
    ptr += nread;
  }
 EXIT:
  return (nbytes-nleft);
}

/* read a line up to nbytes long into character string (newline preserved)*/
int32_t readline2( int filedes, int32_t nbytes, 
		     char ** data, sidl_BaseInterface *_ex ) {

  /* TODO:  This is intentionally a quickly implemented and threadsafe 
            implementation with the downside of being a bit slow and 
	    unsophisticated.  Upgrade to something faster when time allows*/
  ssize_t n, rc;
  char c;
  char *ptr;

  if ( *data == NULL ) { 
    *data = sidl_String_alloc(nbytes);
  }
  ptr = *data;

  for( n=1; n<nbytes;n++) { 
    if (( rc=read(filedes,&c,1))==1) { 
      /* add a character */
      *ptr++=c;
      if (c=='\n') { 
	/* newline is not removed */
	break;       
      }
    } else if (rc == 0 ) { 
      if (n==1) { 
	/* EOF, no data read */
	return 0;  
      } else { 
	/* EOF, some data read */
	break;       
      }
    } else { /* rc = something other than 1 or zero */ 
      if (errno==EINTR) { 
	n--; continue;
      } else { 
	SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "readline() error!");
      }
    }
  }
  return n;
 EXIT:
  return -1;
}

/* write the character string */
int32_t writen2( int filedes, const int32_t nbytes, const char * data, 
		   sidl_BaseInterface *_ex) {
  size_t nleft;
  ssize_t nwritten;
  const char * ptr = data;

  int n = nbytes;
  nleft = n;
  while( nleft > 0 ) { 
    if ( ( nwritten=write(filedes,ptr,nleft))<=0) { 
      if (errno==EINTR) { 
	nwritten=0; /* and call write() again */
      } else { 
	SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "writen() error!");
      }
    }
    nleft -= nwritten;
    ptr += nwritten;
  }
  return (n);
 EXIT:
  return -1;
}

/* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket._includes) */
#line 170 "sidlx_rmi_IPv4Socket_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_IPv4Socket__load(
  void)
{
#line 184 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket._load) */
  /* insert implementation here: sidlx.rmi.IPv4Socket._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket._load) */
#line 190 "sidlx_rmi_IPv4Socket_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_IPv4Socket__ctor(
  /* in */ sidlx_rmi_IPv4Socket self)
{
#line 202 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket._ctor) */
  /*Actually, I don't want to do anything here.  I want data to be null.*/
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket._ctor) */
#line 210 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_IPv4Socket__dtor(
  /* in */ sidlx_rmi_IPv4Socket self)
{
#line 221 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket._dtor) */
  struct sidlx_rmi_IPv4Socket__data * data = sidlx_rmi_IPv4Socket__get_data( self );
  sidl_BaseInterface _ex = NULL;
  if (data) { 
    sidlx_rmi_IPv4Socket_close(self, &_ex); SIDL_CHECK(_ex);/*If the socket isn't closed, close it. Close frees data*/
  }
  return;
 EXIT:
  sidl_BaseInterface_deleteRef(_ex);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket._dtor) */
#line 238 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  getsockname[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_getsockname"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_getsockname(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ sidl_BaseInterface *_ex)
{
#line 250 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.getsockname) */
  struct sockaddr_in localaddr;
  socklen_t len = sizeof(struct sockaddr_in);
  int32_t n = 0;

  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    localaddr.sin_family = AF_INET;
    if ((n=getsockname(dptr->fd, (struct sockaddr*) &localaddr, &len))<0) { 
      SIDL_THROW( *_ex, sidlx_io_IOException, "getsockname() error");
    }
    *port = ntohs(localaddr.sin_port);
    *address = ntohl(localaddr.sin_addr.s_addr);
    return n;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
 EXIT:
  return -1;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.getsockname) */
#line 279 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  getpeername[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_getpeername"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_getpeername(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ sidl_BaseInterface *_ex)
{
#line 289 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.getpeername) */
  struct sockaddr_in localaddr;
  socklen_t len = sizeof(struct sockaddr_in);
  int32_t n = 0;
  
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    localaddr.sin_family = AF_INET;
    if ((n=getpeername(dptr->fd, (struct sockaddr*) &localaddr, &len))<0) { 
      SIDL_THROW( *_ex, sidlx_io_IOException, "getpeername() error");
    }
    *port = ntohs(localaddr.sin_port);
    *address = ntohl(localaddr.sin_addr.s_addr);
    return n;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
 EXIT:
  return -1;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.getpeername) */
#line 321 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  close[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_close"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_close(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 327 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.close) */
  int32_t n = 0;
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    if ((n=close(dptr->fd)) < 0 ) { 
      SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "close() error!");
    }
    free((void *)dptr);
    sidlx_rmi_IPv4Socket__set_data(self, NULL);
    return n;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
 EXIT:
  return -1;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.close) */
#line 356 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  readn[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_readn"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_readn(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 362 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.readn) */
  char* ptr;
  int32_t n_read = 0;
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);

  if (dptr) {
    ensure1DPackedChar( nbytes, data );
    
    ptr = sidl_char__array_first( *data ); /* get the first one */
    
    n_read = readn2(dptr->fd, nbytes, &ptr, _ex ); SIDL_CHECK(*_ex);
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
  return -1;
 EXIT:
  return n_read;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.readn) */
#line 395 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  readline[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_readline"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_readline(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 399 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.readline) */
  char* ptr;
  int32_t n_read = -1;
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);

  if (dptr) {
    ensure1DPackedChar( nbytes, data );
    
    ptr = sidl_char__array_first( *data ); /* get the first one */
    
    n_read = readline2(dptr->fd, nbytes, &ptr, _ex); SIDL_CHECK(*_ex);
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
  return -1;
 EXIT:
  return n_read;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.readline) */
#line 434 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  readstring[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_readstring"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_readstring(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 436 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.readstring) */
  /*I don't really like this.  I think we should have some what communicating back when
    the array was not big enough to handle all the data, so we can handle it later.
    I recommen using s_read_string_alloc intead.*/
  
  char* ptr;
  int32_t bytesToRead = 0;
  int32_t len = sidl_char__array_length(*data,0);
  int32_t n_read = -1;
  int32_t inLen, left;

  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    
    if(nbytes == -1)
      bytesToRead = len;
    else
      left = bytesToRead = (nbytes<len) ? nbytes: len;
    
    ensure1DPackedChar( bytesToRead, data );
    
    ptr = sidl_char__array_first( *data ); /* get the first one */
    n_read = s_readInt(dptr->fd, &inLen, _ex); SIDL_CHECK(*_ex);					   
    if(n_read <= 0 || inLen <= 0) {
      goto EXIT;
    }    

    bytesToRead = (bytesToRead<inLen)? bytesToRead:inLen; 
    /*hopefully bytesToRead is bigger than inLen */
    n_read = readn2(dptr->fd, bytesToRead, &ptr, _ex ); SIDL_CHECK(*_ex);
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
  return -1;
 EXIT:
  return n_read;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.readstring) */
#line 493 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  readstring_alloc[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_readstring_alloc"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_readstring_alloc(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 492 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.readstring_alloc) */
  /* read a string up into character array (newline preserved)
   frees the current sidl_char__array and allocates a new one if length < the incoming string
   if(*data == null) a string as long as nessecary will be allocated */

  int32_t inLen = 0;
  int32_t curLen = 0; 
  int32_t n = 0;
  int32_t lower[1], upper[1];
  
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    if(data ==NULL) {
      SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "read() error: data is NULL!");
      goto EXIT;
    }
    if(*data != NULL)
      curLen = sidl_char__array_length(*data, 0); /*Now that we know data isn't null*/
    else
      curLen = 0;

    n = s_readInt(dptr->fd, &inLen, _ex); SIDL_CHECK(*_ex);
    if(inLen <= 0 || n <= 0) {
      SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "read() error: Length of string <= 0");
      goto EXIT;
    }
    
    if(curLen < inLen) {
      if(*data != NULL)
	sidl_char__array_deleteRef(*data);
      lower[0] = 0;
      upper[0] = inLen-1;
      *data = sidl_char__array_createCol(1,lower,upper);
    }
    
    n = s_readn(dptr->fd, inLen, data, _ex); SIDL_CHECK(*_ex);
    return n;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
  EXIT:
  return -1;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.readstring_alloc) */
#line 557 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  readint[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_readint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_readint(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout */ int32_t* data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 554 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.readint) */
  int32_t ret = 0;
  char ** cast_data = (char**) &data;
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    ret = readn2(dptr->fd, 4, cast_data, _ex); SIDL_CHECK(*_ex);
    *data = ntohl(*data);
    return ret;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
 EXIT:
  return 0;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.readint) */
#line 591 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  writen[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_writen"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_writen(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 587 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.writen) */
  char * ptr= sidl_char__array_first( data );
  int32_t n = sidl_char__array_length(data,0);
  int32_t n_written;
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    if (nbytes != -1 ) { 
      /* unless nbytes is -1 (meaning "all"), take the min(nbytes, length()) */
      n = (n<nbytes)? n : nbytes;
    }
    n_written = writen2( dptr->fd, n, ptr, _ex); SIDL_CHECK(*_ex);
    return n_written;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
 EXIT:
  return -1;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.writen) */
#line 630 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  writestring[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_writestring"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_writestring(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 624 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.writestring) */
  char * ptr= sidl_char__array_first( data );
  int32_t n = sidl_char__array_length(data,0);
  int32_t n_written;
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    if (nbytes != -1 ) { 
      /* unless nbytes is -1 (meaning "all"), take the min(nbytes, length()) */
      n = (n<nbytes)? n : nbytes;
    }
    s_writeInt(dptr->fd, n, _ex); SIDL_CHECK(*_ex);
    n_written = writen2( dptr->fd, n, ptr, _ex); SIDL_CHECK(*_ex);
    return n_written;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
 EXIT:
  return -1;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.writestring) */
#line 670 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  writeint[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_writeint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_writeint(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 661 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.writeint) */
  /* Insert-Code-Here {sidlx.rmi.IPv4Socket.writeint} (writeint method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.writeint) */
#line 693 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  setFileDescriptor[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_setFileDescriptor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_IPv4Socket_setFileDescriptor(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t fd,
  /* out */ sidl_BaseInterface *_ex)
{
#line 682 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.setFileDescriptor) */
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    dptr->fd = fd;
  } else {
    dptr = malloc(sizeof(struct sidlx_rmi_IPv4Socket__data));
    dptr->fd = fd;
  }
  sidlx_rmi_IPv4Socket__set_data(self, dptr);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.setFileDescriptor) */
#line 724 "sidlx_rmi_IPv4Socket_Impl.c"
}

/*
 * Method:  getFileDescriptor[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_IPv4Socket_getFileDescriptor"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_IPv4Socket_getFileDescriptor(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 710 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket.getFileDescriptor) */
  struct sidlx_rmi_IPv4Socket__data *dptr =
    sidlx_rmi_IPv4Socket__get_data(self);
  if (dptr) {
    return dptr->fd;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "This Socket isn't initialized!");
 EXIT:
  return -1;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket.getFileDescriptor) */
#line 753 "sidlx_rmi_IPv4Socket_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_IPv4Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj) {
  return sidlx_rmi_IPv4Socket__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
