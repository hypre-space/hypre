
#include <errno.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include "sidlx_common.h"
#include "sidl_String.h"
#include "sidl_rmi_NetworkException.h"
#include "sidl_rmi_MalformedURLException.h"
#include "sidl_io_IOException.h"

int s_socket( int family, int type, int protocol, sidl_BaseInterface *_ex) { 
  int n;
  
/* socket() returns nonnegative descriptor if ok, -1 on error */
  if ((n=socket(family,type,protocol))< 0) { 
    SIDL_THROW( *_ex, sidl_io_IOException, "socket() error");
  }
 EXIT:
  return(n);
}

int s_bind( int sockfd, const struct sockaddr * myaddr, socklen_t addrlen, 
	    sidl_BaseInterface *_ex ) { 
  int n;

  /* bind() returns 0 if ok, -1 on error */
  if ((n=bind(sockfd, myaddr, addrlen)) < 0 ) { 
    SIDL_THROW( *_ex, sidl_io_IOException, "bind() error");
  }
 EXIT:
  return n; 
}


int s_listen( int sockfd, int backlog, sidl_BaseInterface *_ex) { 
  int n;
  
  /* listen() returns 0 if ok, -1 on error */
  if ((n=listen(sockfd,backlog))<0) { 
    SIDL_THROW( *_ex, sidl_io_IOException, "listen() error");
  }
 EXIT:
  return n;
}

int s_accept( int sockfd, struct sockaddr *cliaddr, socklen_t *addrlen, 
	    sidl_BaseInterface *_ex ) { 
  int n;
  
  /* accept() returns nonnegative descriptor if OK, -1 on error */
  if ((n=accept(sockfd, cliaddr, addrlen)) < 0 ) { 
    SIDL_THROW( *_ex, sidl_io_IOException, "accept() error");
  }
 EXIT:
  return n;
}

static int
isRecoverable(int errval)
{
  switch(errval) {
  case ENOMEM:
  case ENOBUFS:
  case ENFILE:
  case EMFILE:
  case ECONNABORTED:
  case EAGAIN:
  case EINTR:
  case ETIMEDOUT:
  case ETIME:
    return 1;
  }
  return 0;
}

int
s_connect( int sockfd, const struct sockaddr *servaddr, socklen_t addrlen,
           sidl_BaseInterface *_ex ) { 
  static long int s_connect_retries = 1;
  static int s_env_check = 0;
  unsigned long sleeptime = 512; /* around half a millisecond */
  long int i;
  int n = -1;
  if (!s_env_check) {
    char *val;
    if ((val = getenv("SIDLX_CONNECT_MAX_RETRIES"))) {
      long int newval = strtol(val, NULL, 0);
      if ((newval != LONG_MIN) && (newval != LONG_MAX)) {
        s_connect_retries = ((newval >= 0) ? newval : 0);
      }
    }
    s_env_check = 1;
  }
  
  for(i = 0; i <= s_connect_retries; ++i) {
    /* returns 0 if OK, -1 on error */
    if ((n = connect( sockfd, servaddr, addrlen )) == 0) break;
    else if (isRecoverable(errno) && (i < s_connect_retries)) {
      if (sleeptime >= 150000000UL) {
        sleeptime = 300000000UL;  /* maximum sleep of 5 minutes! */
      }
      else {
        sleeptime <<= 1;          /* double sleep */
      }
      usleep(sleeptime);
    }
  }
  if (n < 0) {
    SIDL_THROW( *_ex, sidl_io_IOException, "connect() error");
  }
 EXIT:
  return n;
}


pid_t s_fork( sidl_BaseInterface *_ex ) { 
  
  pid_t pid;
  
  /* returns 0 in child, PID in parent, -1 on error */
  if ((pid=fork()) < 0) { 
    SIDL_THROW( *_ex, sidl_io_IOException, "fork() error");
  }
 EXIT:
  return pid;
}


int s_close( int sockfd, sidl_BaseInterface *_ex) { 
  int n;
  
  /* returns 0 if okay, -1 on error */
  if ((n=close(sockfd)) < 0 ) { 
    SIDL_THROW( *_ex, sidl_io_IOException, "close() error");
  }
 EXIT:
  return n;
}


/* same as getsockname(), but using sidl exceptions */
int s_getsockname( int sockfd, struct sockaddr *localaddr, socklen_t *addrlen, 
		   sidl_BaseInterface *_ex) { 
  int n;
  
  /* returns 0 if OK, -1 on error */
  if ((n=getsockname(sockfd, localaddr, addrlen))<0) { 
    SIDL_THROW( *_ex, sidl_io_IOException, "getsockname() error");
  }
 EXIT:
  return n;
}

/* same as getpeername(), but using sidl exceptions */
int s_getpeername( int sockfd, struct sockaddr *peeraddr, socklen_t *addrlen, 
		 sidl_BaseInterface *_ex) { 
  int n;
  
  /* returns 0 if OK, -1 on error */
  if ((n=getpeername(sockfd, peeraddr, addrlen))<0) { 
    SIDL_THROW( *_ex, sidl_io_IOException, "getpeername() error");
  }
 EXIT:
  return n;
}


/* this is a utility function that makes sure a char array is
   1-D, packed, and has a minimum length. */
void ensure1DPackedCharArray( const int32_t minlen, 
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

/* read nbytes into a character array */
int32_t s_readn( int filedes, int32_t nbytes, struct sidl_char__array** data,
		 sidl_BaseInterface *_ex) {
  char* ptr;
  int32_t n_read;

  ensure1DPackedCharArray( nbytes, data );

  ptr = sidl_char__array_first( *data ); /* get the first one */
					   
  n_read = s_readn2(filedes, nbytes, &ptr, _ex ); SIDL_CHECK(*_ex);
 EXIT:
  return n_read;
}

/* read a line up to nbytes long into character array (newline preserved)*/
int32_t s_readline( int filedes, int32_t nbytes, 
		    struct sidl_char__array** data, sidl_BaseInterface *_ex ) {
  char* ptr;
  int32_t n_read;

  ensure1DPackedCharArray( nbytes, data );

  ptr = sidl_char__array_first( *data ); /* get the first one */
  
  n_read = s_readline2(filedes, nbytes, &ptr, _ex); SIDL_CHECK(*_ex);

  return n_read;
 EXIT:
  return -1;
}

/* write nbytes of a character array (-1 implies whole array) */
int32_t s_writen( int filedes, int32_t nbytes, 
		  struct sidl_char__array* data, sidl_BaseInterface *_ex) {
  char * ptr= sidl_char__array_first( data );
  int32_t n = sidl_char__array_length(data,0);
  int32_t n_written;
  if (nbytes != -1 ) { 
    /* unless nbytes is -1 (meaning "all"), take the min(nbytes, length()) */
    n = (n<nbytes)? n : nbytes;
  }
  n_written = s_writen2( filedes, n, ptr, _ex); SIDL_CHECK(*_ex);
  return n_written;
 EXIT:
  return -1;
}

/* read a null terminated string from a FILE */
int32_t s_fgets( FILE * fp, const int32_t maxlen, struct sidl_char__array ** data, sidl_BaseInterface *_ex ) { 
  char * p;
  char * ptr;

  ensure1DPackedCharArray( maxlen, data );
  
  ptr= sidl_char__array_first(* data );
  p = fgets( ptr, maxlen+1, fp );

  if ( p==NULL ) { 
    return 0;
  } else { 
    return strlen(p);
  }
}

/* write a null terminated string to a FILE */
int32_t s_fputs( FILE *fp, const int32_t nbytes, 
		 const struct sidl_char__array * data, 
		 sidl_BaseInterface *_ex ) { 
  char * ptr; 
  int n; 

  if (data == NULL || (sidl_char__array_dimen(data)!=1) || (sidl_char__array_stride(data,0)!=1) ) { 
    return -1;
  }

  ptr= sidl_char__array_first( data );
  n = sidl_char__array_length(data,0)-1;
  ptr[n]='\0'; /* just to be safe */
  if(nbytes != -1) {
    if(nbytes < n)
      ptr[nbytes-1]='\0';
  }

  return fputs( ptr, fp );
}

/* read an int32_t from the network */
int32_t s_readInt(int filedes, int32_t* data, sidl_BaseInterface *_ex) {
  int32_t ret = 0;
  char ** cast_data = (char**) &data;
  
  ret = s_readn2(filedes, 4, cast_data, _ex); SIDL_CHECK(*_ex);
  *data = ntohl(*data);
  return ret;
 EXIT:
  return 0;
}

/* read nbytes into a character string */
int32_t s_readn2( int filedes, int32_t nbytes, char ** data,
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
	SIDL_THROW( *_ex, sidl_io_IOException, "read() error");
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
int32_t s_readline2( int filedes, int32_t nbytes, 
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
	SIDL_THROW( *_ex, sidl_io_IOException, "read() error");
      }
    }
  }
  return n;
 EXIT:
  return -1;
}

/*write an int32_t to the network*/
void s_writeInt(int filedes, int32_t data, sidl_BaseInterface *_ex) {
  
  char * cast_data = (char*) &data;
  data = htonl(data);
  s_writen2(filedes, 4, cast_data, _ex);

}

/* write the character string */
int32_t s_writen2( int filedes, const int32_t nbytes, const char * data, 
		   sidl_BaseInterface *_ex) {
  size_t nleft;
  ssize_t nwritten;
  const char * ptr = data;
  /* int n = sidl_String_strlen( data ); */
  int n = nbytes;
  nleft = n;
  while( nleft > 0 ) { 
    if ( ( nwritten=write(filedes,ptr,nleft))<=0) { 
      if (errno==EINTR) { 
	nwritten=0; /* and call write() again */
      } else { 
	SIDL_THROW( *_ex, sidl_io_IOException, "write() error");
      }
    }
    nleft -= nwritten;
    ptr += nwritten;
  }
  return (n);
 EXIT:
  return -1;
}

/* write nbytes of this character array as a string.  (an length integer 
   followed by the byte stream) -1 means write the whole string*/ 
int32_t s_write_string(int filedes, const int32_t nbytes, 
		       struct sidl_char__array * data, 
		       sidl_BaseInterface *_ex) {

  char * ptr= sidl_char__array_first( data );
  int32_t n = sidl_char__array_length(data,0);
  int32_t n_written;
  if (nbytes != -1 ) { 
    /* unless nbytes is -1 (meaning "all"), take the min(nbytes, length()) */
    n = (n<nbytes)? n : nbytes;
  }
  s_writeInt(filedes, n, _ex); SIDL_CHECK(*_ex);
  n_written = s_writen2( filedes, n, ptr, _ex); SIDL_CHECK(*_ex);
  return n_written;
 EXIT:
  return -1;

}
/*
I don't really like this.  I think we should have some what communicating back when
the array was not big enough to handle all the data, so we can handle it later.
I recommen using s_read_string_alloc intead.
read a string up to min(nbytes,length) long into character array (newline preserved)
*/
int32_t s_read_string( int filedes, const int32_t nbytes, 
		       struct sidl_char__array* data, sidl_BaseInterface *_ex ) {
  char* ptr;
  int32_t bytesToRead = 0;
  int32_t len = sidl_char__array_length(data,0);
  int32_t n_read, inLen, left;

  if(nbytes == -1)
    bytesToRead = len;
  else
    left = bytesToRead = (nbytes<len) ? nbytes: len;

  ensure1DPackedCharArray( bytesToRead, &data );

  ptr = sidl_char__array_first( data ); /* get the first one */
  n_read = s_readInt(filedes, &inLen, _ex); SIDL_CHECK(*_ex);					   
  if(n_read == 0)
    goto EXIT;
  
  bytesToRead = (bytesToRead<inLen)? bytesToRead:inLen; 
  /* hopefully bytesToRead is bigger than inLen */
  n_read = s_readn2(filedes, bytesToRead, &ptr, _ex ); SIDL_CHECK(*_ex);
  
 EXIT:
  return n_read;

}

/* read a string up into character array (newline preserved)
   frees the current sidl_char__array and allocates a new one if length < the incoming string
   if(*data == null) a string as long as nessecary will be allocated */
int32_t s_read_string_alloc( int filedes,
			     struct sidl_char__array** data, sidl_BaseInterface *_ex ) {

  int32_t inLen = 0;
  int32_t curLen = 0; 
  int32_t n;
  int32_t lower[1], upper[1];
  if(data ==NULL) {
    SIDL_THROW( *_ex, sidl_io_IOException, "read() error: data is NULL!");
    return 0;
  }
  if(*data != NULL)
    curLen = sidl_char__array_length(*data, 0); /* Now that we know data isn't null */
  else
    curLen = 0;

  n = s_readInt(filedes, &inLen, _ex); SIDL_CHECK(*_ex);
  if(inLen <= 0) {
    SIDL_THROW( *_ex, sidl_io_IOException, "read() error: Length of string <= 0");
    return 0;
  }

  if(curLen < inLen) {
    if(*data != NULL)
      sidl_char__array_deleteRef(*data);
    lower[0] = 0;
    upper[0] = inLen-1;
    *data = sidl_char__array_createCol(1,lower,upper);
  }

  n = s_readn(filedes, inLen, data, _ex); SIDL_CHECK(*_ex);
  return n;

  EXIT:
  return 0;

}

/* This function parses a url into the pointers provided (they are all out parameters)
   url, protocol, and server are required, and the method will throw an if they are
   null.  port, className, and objectID are optional, and may be passed in as NULL
   Support Port Ranges.  If a range is not detected, start_port has the real port
*/ 
void sidlx_parseURL(const char* url, char** protocol, char** server, 
		    int* start_port, int* end_port, 
		    char** objectID, sidl_BaseInterface *_ex) {

  int i = 0;
  int start=0;
  int length = 0;

  if(url == NULL || protocol == NULL || server == NULL) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, 
	       "sidl_rmi_ProtocolFactory.praseURL: Required arg is NULL\n");
  }
  
  length = sidl_String_strlen(url);
  
  /* extract the protocol name */
  while ((i<length) && (url[i]!=':')) { 
    i++;
  }
  if ( (i==start) || (i==length) ) { 
    SIDL_THROW(*_ex, sidl_rmi_MalformedURLException, "could not extract prefix from URL\n");
  }
  
  if(protocol != NULL) {
    *protocol=sidl_String_strndup(url,i); /* copies (i-1) chars plus a '\0'*/
  }

  /* skip colons & slashes (should be "://") */
  if ( ((i+3)>=length) || (url[i]!=':') || (url[i+1]!='/') || (url[i+2]!='/')) { 
    SIDL_THROW(*_ex, sidl_rmi_MalformedURLException, "expected :// next in URL\n");
  } else { 
    i+=3;
  }
  /* extract server name */
  start=i;
  while ( (i<length) && url[i]!=':'&& url[i]!='/') { 
    i++;
  }

  if (i==start) { 
    SIDL_THROW(*_ex, sidl_rmi_MalformedURLException, "could not extract host from URL");
  }
  if(server != NULL) {
    *server = sidl_String_strndup(url + start, i-start);
  }

  /* extract port number (if it exists ) */
  if ( (i<length) && (url[i]==':')) {
    ++i;
    start=i;
    while ((i<length) && (url[i] != '/') && (url[i] != '-')) { 
      if ( (url[i]<'0') || url[i]>'9') { 
	SIDL_THROW(*_ex, sidl_rmi_MalformedURLException, "could not extract port number from URL");
      }
      i++;
    }
    if(start_port!=NULL) {
      char buffer[256];
      strncpy( buffer, url+start, i-start );
      buffer[i-start] = '\0';
      *start_port = atoi( buffer );
    }
  }

  /* extract end_port number (if a range exists exists ) */
  if ( (i<length) && (url[i]=='-')) {
    ++i;
    start=i;
    while ((i<length) && (url[i] != '/')) { 
      if ( (url[i]<'0') || url[i]>'9') { 
	SIDL_THROW(*_ex, sidl_rmi_MalformedURLException, "could not extract max port from URL\n");
      }
      i++;
    }
    if(end_port!=NULL) {
      char buffer[256];
      strncpy ( buffer, url+start, i-start);
      *end_port = atoi( buffer );
    }
  } else {  /*If an end_port was requested, but isn't in the URL, end_port =
              0; */
    if(end_port!=NULL) {
      *end_port = 0;
    }
  }

  /* Continue onward to extract the objectid, if it exists*/
  if ( (i<length) && (url[i]=='/')) {
    ++i;
    start=i;
    while ((i<length) && (url[i] != '/')) { 
      i++;
    }
    if(objectID!=NULL) {
      *objectID = sidl_String_strndup( url+start,i-start );
    }
  } else {
    return;
  }

 EXIT:
  return;

}
