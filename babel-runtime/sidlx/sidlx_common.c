


#include "sidlx_common.h"

int s_socket( int family, int type, int protocol, sidl_BaseInterface *_ex) { 
  int n;
  
/* socket() returns nonnegative descriptor if ok, -1 on error */
  if ((n=socket(family,type,protocol))< 0) { 
    SIDL_THROW( *_ex, sidlx_io_IOException, "socket() error");
  }
 EXIT:
  return(n);
}

int s_bind( int sockfd, const struct sockaddr * myaddr, socklen_t addrlen, 
	    sidl_BaseInterface *_ex ) { 
  int n;

  /* bind() returns 0 if ok, -1 on error */
  if ((n=bind(sockfd, myaddr, addrlen)) < 0 ) { 
    SIDL_THROW( *_ex, sidlx_io_IOException, "bind() error");
  }
 EXIT:
  return n; 
}


int s_listen( int sockfd, int backlog, sidl_BaseInterface *_ex) { 
  int n;
  
  /* listen() returns 0 if ok, -1 on error */
  if ((n=listen(sockfd,backlog))<0) { 
    SIDL_THROW( *_ex, sidlx_io_IOException, "listen() error");
  }
 EXIT:
  return n;
}

int s_accept( int sockfd, struct sockaddr *cliaddr, socklen_t *addrlen, 
	    sidl_BaseInterface *_ex ) { 
  int n;
  
  /* accept() returns nonnegative descriptor if OK, -1 on error */
  if ((n=accept(sockfd, cliaddr, addrlen)) < 0 ) { 
    SIDL_THROW( *_ex, sidlx_io_IOException, "accept() error");
  }
 EXIT:
  return n;
}


int s_connect( int sockfd, const struct sockaddr *servaddr, socklen_t addrlen,
	      sidl_BaseInterface *_ex ) { 
  int n;
  
  /* returns 0 if OK, -1 on error */
  if ((n = connect( sockfd, servaddr, addrlen )) < 0 ) { 
    SIDL_THROW( *_ex, sidlx_io_IOException, "connect() error");
  }
 EXIT:
  return n;
}


pid_t s_fork( sidl_BaseInterface *_ex ) { 
  
  pid_t pid;
  
  /* returns 0 in child, PID in parent, -1 on error */
  if ((pid=fork()) < 0) { 
    SIDL_THROW( *_ex, sidlx_io_IOException, "fork() error");
  }
 EXIT:
  return pid;
}


int s_close( int sockfd, sidl_BaseInterface *_ex) { 
  int n;
  
  /* returns 0 if okay, -1 on error */
  if ((n=close(sockfd)) < 0 ) { 
    SIDL_THROW( *_ex, sidlx_io_IOException, "close() error");
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
    SIDL_THROW( *_ex, sidlx_io_IOException, "getsockname() error");
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
    SIDL_THROW( *_ex, sidlx_io_IOException, "getpeername() error");
  }
 EXIT:
  return n;
}

/* read nbytes into a character array */
int32_t s_readn( int filedes, int32_t nbytes, struct sidl_char__array** data,
		 sidl_BaseInterface *_ex) {
  size_t nleft;
  ssize_t nread;
  char* ptr;
  int fd;

  *data = sidl_char__array_create1d(nbytes);

  ptr= sidl_char__array_first( *data );
  nleft = nbytes;
  
  while ( nleft > 0 ) {
    if ( ( nread = read( filedes, ptr, nleft)) < 0 ) { 
      if ( errno == EINTR ) { 
	nread = 0; /* and call read() again */
      } else { 
	nleft = nbytes+1;
	SIDL_THROW( *_ex, sidlx_io_IOException, "read() error");
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

/* read a line up to nbytes long into character array (newline preserved)*/
int32_t s_readline( int filedes, int32_t nbytes, 
		    struct sidl_char__array** data, sidl_BaseInterface *_ex ) {

  /* NOTE:  This is intentionally a quickly implemented and threadsafe 
            implementation with the downside of being a bit slow and 
	    unsophisticated.  Upgrade to something faster when time allows*/
  ssize_t n, rc;
  char c;
  char *ptr;
  struct sidl_char__array* buffer = NULL;
  
  buffer = sidl_char__array_create1d(nbytes);
  ptr = sidl_char__array_first(buffer);

  for( n=1; n<nbytes;n++) { 
    if (( rc=read(filedes,&c,1))==1) { 
      /* add a character */
      *ptr++=c;
      if (c='\n') { 
	/* newline is not removed */
	break;       
      }
    } else if (rc == 0 ) { 
      if (n==1) { 
	/* EOF, no data read */
	sidl_char__array_deleteRef(buffer);
	buffer=NULL;
	return (0);  
      } else { 
	/* EOF, some data read */
	break;       
      }
    } else { 
      if (errno==EINTR) { 
	n--; continue;
      } else { 
	sidl_char__array_deleteRef(buffer);
	buffer=NULL;
	SIDL_THROW( *_ex, sidlx_io_IOException, "read() error");
      }
    }
  }


  if (*data != NULL && n==sidl_char__array_upper(*data,0) ) { 
    strncpy( sidl_char__array_first(*data), sidl_char__array_first(buffer), n);
  } else { 
    if (*data != NULL ) { 
      sidl_char__array_deleteRef( *data );
      *data=NULL;
    }
    *data = sidl_char__array_slice(buffer,1,&n,NULL,NULL,NULL);
  }
  if( buffer ) { 
    sidl_char__array_deleteRef(buffer);
    buffer=NULL;
  }
  return n;
 EXIT:
  if( buffer ) { 
    sidl_char__array_deleteRef(buffer);
    buffer=NULL;
  }
  return -1;
}

/* write the character array */
int32_t s_writen( int filedes, struct sidl_char__array* data, sidl_BaseInterface *_ex) {
  size_t nleft;
  ssize_t nwritten;
  const char * ptr;
  int n;

  /* assert( sidl_char__array_isRowOrder(data)||sidl_char_array_isColOrder(data)) */
  ptr= sidl_char__array_first( data );
  n = sidl_char__array_upper(data,0)-sidl_char__array_lower(data,0);
  nleft = n;
  while( nleft > 0 ) { 
    if ( ( nwritten=write(filedes,ptr,nleft))<=0) { 
      if (errno==EINTR) { 
	nwritten=0; /* and call write() again */
      } else { 
	SIDL_THROW( *_ex, sidlx_io_IOException, "write() error");
      }
    }
    nleft -= nwritten;
    ptr += nwritten;
  }
  return (n);
 EXIT:
  return -1;
}



/* read a null terminated string from a FILE */
int32_t s_fgets( FILE * fp, int32_t maxlen, struct sidl_char__array ** data, sidl_BaseInterface *_ex ) { 
  char * p;
  char * ptr;
  int realloc = 1;
  int n; 

  if (*data != NULL) { 
    if ((sidl_char__array_dimen(*data)==1) || (sidl_char__array_stride(*data,0)==1) ) { 
      n = sidl_char__array_upper(*data,0)-sidl_char__array_lower(*data,0);
      if ( n>= maxlen ) { 
	realloc = 0;
      }
    }
    if (realloc) { 
      sidl_char__array_deleteRef(*data);
      *data=NULL;
    }
  }
  if (*data==NULL) { 
    *data = sidl_char__array_create1d(maxlen+1);
  }
  
  ptr= sidl_char__array_first(* data );
  p = fgets( ptr, maxlen+1, fp );

  if ( p==NULL ) { 
    return 0;
  } else { 
    return strlen(p);
  }
}

/* write a null terminated string to a FILE */
int32_t s_fputs( FILE *fp, struct sidl_char__array * data, sidl_BaseInterface *_ex ) { 
  char * ptr; 
  int n; 

  if (data == NULL || (sidl_char__array_dimen(data)!=1) || (sidl_char__array_stride(data,0)!=1) ) { 
    return -1;
  }

  ptr= sidl_char__array_first( data );
  n = sidl_char__array_upper(data,0)-sidl_char__array_lower(data,0);

  ptr[n]='\0'; /* just to be safe */

  return fputs( ptr, fp );
}
