#include <errno.h>       /* obligatory includes */
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <iostream.h>
#include <string.h>

#define MAXHOSTNAME 128

extern "C" void fireman(int);
void do_something(int);

//============================================================================

void *memset(void *s, int c, size_t n);

//============================================================================
int establish(unsigned short portnum){
  char   myname[MAXHOSTNAME+1];
  int    s;
  struct sockaddr_in sa;
  struct hostent *hp;

  memset(&sa, 0, sizeof(struct sockaddr_in)); /* clear our address */
  gethostname(myname, MAXHOSTNAME);           /* who are we? */
  hp= gethostbyname(myname);                  /* get our address info */
  if (hp == NULL)                             /* we don't exist !? */
    return(-1);
  sa.sin_family= hp->h_addrtype;              /* this is our host address */
  sa.sin_port= htons(portnum);                /* this is our port number */
  if ((s= socket(AF_INET, SOCK_STREAM, 0)) < 0) /* create socket */
    return(-1);
  if (bind(s,(const sockaddr*)&sa,sizeof(struct sockaddr_in)) < 0) {
    close(s);
    return(-1);                               /* bind address to socket */
  }
  listen(s, 1);                               /* max # of queued connects */
  return(s);
}

//============================================================================

int get_connection(int s){
  int t;                  /* socket of connection */

  if ((t = accept(s,NULL,NULL)) < 0)   /* accept connection if there is one */
    return(-1);
  return(t);
}

//============================================================================

int read_data(int s,     /* connected socket */
              char *buf, /* pointer to the buffer */
              int n      /* number of characters (bytes) we want */
             ){ 
  int bcount; /* counts bytes read */
  int br;     /* bytes read this pass */

  bcount= 0;
  br= 0;
  while (bcount < n) {             /* loop until full buffer */
    if ((br= read(s,buf,n-bcount)) > 0) {
      bcount += br;                /* increment byte counter */
      buf += br;                   /* move buffer ptr for next read */
    }
    else if (br < 0)               /* signal an error to the caller */
      return(-1);
    if (buf[-1] == '*')
      break;
  }
  return(bcount);
}


//============================================================================

int read_socket(int portnum, char *Buf){
  int s = 0, t = 0;

  if ((s = establish(portnum)) < 0)  /* plug in the phone */
    return 0;

  if ((t= get_connection(s)) < 0)    /* get a connection */
    if (errno != EINTR)              /* EINTR might happen on accept(), */
      return 0;

  close(s);
  
  char *b = Buf;
  read_data(t, b, 1024);
  
  close(t);
  return 1;
}

//============================================================================
// Compile with :
// CC -o toni server.cpp -lsocket -lnsl
//============================================================================
