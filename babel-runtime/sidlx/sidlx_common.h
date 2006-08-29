/*
 * Commonly shared stuff used throughout sidlx
 */
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "sidl_header.h"
#include "sidl_Exception.h" /* macros for try-catch equivalents */
#include "sidl_BaseInterface.h"




#define MAXSOCKADDR 128
#define MAXLINE 1023

#ifdef __cplusplus
extern "C" { /* } */
#endif 

  /* 
   * A handy macro for mallocing and setting data 
   * Use this inside a _ctor method.
   */
#define SIDLX_MALLOC_SET_DATA( PKG_CLASS )		\
  struct PKG_CLASS ## __data *data =			\
    (struct PKG_CLASS ## __data *)			\
       malloc( sizeof (  struct PKG_CLASS ## __data) ); \
    PKG_CLASS ## __set_data(self,data);

  /* 
   * A handy macro for deleting and unsetting data.
   * Typically used at the very end of a _dtor() method
   */
#define SIDLX_FREE_UNSET_DATA( PKG_CLASS )				\
  struct PKG_CLASS ## __data *data = PKG_CLASS ## __get_data(self);	\
    if (data) {								\
      free((void*)data);						\
    }									\
    PKG_CLASS ## __set_data(self,NULL);


  /* A struct guaranteed to be large enough for any sockaddr */

  typedef union { 
    struct sockaddr sa;
    char data[MAXSOCKADDR];
  } anysockaddr;

  /* same a socket(), but using sidl exceptions */
  int s_socket( int family, int type, int protocol, 
		sidl_BaseInterface *_ex );

  /* same as bind(), but using sidl exceptions */
  int s_bind( int sockfd, const struct sockaddr * myaddr, socklen_t addrlen, 
	      sidl_BaseInterface *_ex );

  /* same as listen(), but using sidl exceptions */
  int s_listen( int sockfd, int backlog, sidl_BaseInterface *_ex);

  /* same as accept(), but using sidl exceptions */
  int s_accept( int sockfd, struct sockaddr *cliaddr, socklen_t *addrlen, 
		sidl_BaseInterface *_ex );

  /* same as fork(), but using sidl exceptions */
  pid_t s_fork( sidl_BaseInterface *_ex );

  /* same as close(), but using sidl exceptions */
  int s_close( int sockfd, sidl_BaseInterface *_ex);

  /* same as getsockname(), but using sidl exceptions */
  int s_getsockname( int sockfd, struct sockaddr *localaddr, socklen_t *addrlen, 
		     sidl_BaseInterface *_ex);

  /* same as getpeername(), but using sidl exceptions */
  int s_getpeername( int sockfd, struct sockaddr *peeraddr, socklen_t *addrlen, 
		     sidl_BaseInterface *_ex);


  /* read nbytes into a character array */
  int32_t s_readn( int filedes, const int32_t nbytes, 
		   struct sidl_char__array** data,
		   sidl_BaseInterface *_ex);

  /* read a line up to nbytes long into character array (newline preserved)*/
  int32_t s_readline( int filedes, const int32_t nbytes, 
		      struct sidl_char__array** data, sidl_BaseInterface *_ex );

  /* write nbytes of a character array (-1 implies whole array) */
  int32_t s_writen( int filedes, const int32_t nbytes, 
		    struct sidl_char__array * data, 
		    sidl_BaseInterface *_ex);

  /* read a null terminated string from a FILE (returns length) */
  int32_t s_fgets( FILE * fp, const int32_t maxlen, struct sidl_char__array ** data, sidl_BaseInterface *_ex );

  /* write a null terminated string to a FILE */
  int32_t s_fputs( FILE *fp, const int32_t nbytes, 
		   const struct sidl_char__array * data, 
		   sidl_BaseInterface *_ex );

  /* In the following routines, the data is put in a char string
     If *data != NULL, then it is assumed the buffer is of sufficient size
     If *data == NULL, then a SIDL_String_malloc() is used to allocate space 
  */
  /* read an int32_t from the network */
  int32_t s_readInt(int filedes, int32_t* data,sidl_BaseInterface *_ex);

  /* read nbytes into a character string */
  int32_t s_readn2( int filedes, const int32_t nbytes, char ** data, 
		    sidl_BaseInterface *_ex);

  /* read a line up to nbytes long into character string (newline preserved)*/
  int32_t s_readline2( int filedes, const int32_t nbytes, 
		       char ** data, sidl_BaseInterface *_ex );
  /*write an int32_t to the network*/
  void s_writeInt(int filedes, const int32_t data, sidl_BaseInterface *_ex);

  /* write the character string */
  int32_t s_writen2( int filedes, const int32_t nbytes, const char * data, 
		     sidl_BaseInterface *_ex);

  /* write nbytes of this character array as a string.  (an length integer 
     followed by the byte stream) -1 means write the whole string*/ 
  int32_t s_write_string(int filedes, const int32_t nbytes, 
			 struct sidl_char__array * data, 
			 sidl_BaseInterface *_ex);

  /* read a string up to min(nbytes,length) long into character array (newline preserved)
     returns the length of the string or readn error code
     nbytes == -1 makes nbytes ignored*/
  int32_t s_read_string( int filedes, const int32_t nbytes, 
			 struct sidl_char__array* data, sidl_BaseInterface *_ex );

  /* read a string up to nbytes long into character array (newline preserved)
     frees the current sidl_char__array and allocates a new one if length < nbytes
     if(nbytes == -1) a string as long as nessecary will be allocated */
  int32_t s_read_string_alloc( int filedes,
			       struct sidl_char__array** data, sidl_BaseInterface *_ex );
  
  /* This function parses a url into the pointers provided (they are all out parameters)
     url, protocol, and server are required, and the method will throw an if they are
     null.  start_port, end_port, className, and objectID are optional, and may be passed in as NULL
     start_port and end_port allow port ranges, such as simhandle://localhost:9000-9999/ 
     If there is no range, the single port comes back in start_port and end_port is 0.
     They are also no gauranteed to be in acending order, they may need to be flipped.
  */ 
  void sidlx_parseURL(const char* url, char** protocol, char** server, 
		      int* start_port, int* end_port, 
		      char** objectID, sidl_BaseInterface *_ex);
  
#ifdef __cplusplus
  /*extern "C" {*/  }
#endif 

