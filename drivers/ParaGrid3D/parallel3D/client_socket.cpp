#include <netdb.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <iostream.h>
#include <unistd.h>
#include <stdio.h>
#include "definitions.h"


//============================================================================
// After getting a socket to make the call with, and giving it an address, 
// you use the connect() function to try to connect to a listening socket.
// The following function calls a particular port number on a particular host:
//============================================================================
int call_socket(char *hostname, unsigned short portnum){ 
  struct sockaddr_in  sa;
  struct hostent     *hp;
  int a, s;

  if ((hp= gethostbyname(hostname)) == NULL) { /* do we know the host's */
    printf("Unknown address\n");               /* address? */
    return(-1);                                /* no */
  }

  memset(&sa,0,sizeof(sa));
  memcpy((char *)&sa.sin_addr,hp->h_addr,hp->h_length); /* set address */
  sa.sin_family= hp->h_addrtype;
  sa.sin_port= htons((u_short)portnum);

  if ((s= socket(hp->h_addrtype,SOCK_STREAM,0)) < 0)   /* get socket */
    return(-1);
  if (connect(s,(const sockaddr*)&sa,sizeof sa) < 0) { /* connect */
    close(s);
    return(-1);
  }
  return(s);
}

//============================================================================

void send_through_socket( int s, char *Buf, int size){
  int bcount; /* counts bytes read */
  int br;     /* bytes read this pass */
  char *b = Buf;

  bcount =   0;
  br     =   0;
  b      = Buf;
  while (bcount < size) {
    if ((br= write(s,b,size-bcount)) > 0) {
      bcount += br;   
      b += br;        
    }
    else if (br < 0){ 
      cout << "\nOn your local machine start mcgl.sunos  with : \n"
	   << "mcgl.sunos -Mci 5000 \n\n";
      exit(1);
      // break;
    }
  }
  close(s);
}

//============================================================================
