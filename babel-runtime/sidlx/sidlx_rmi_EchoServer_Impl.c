/*
 * File:          sidlx_rmi_EchoServer_Impl.c
 * Symbol:        sidlx.rmi.EchoServer-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.rmi.EchoServer
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.6
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.EchoServer" (version 0.1)
 * 
 * Echos the string back to the client... useful for network debuggin
 */

#include "sidlx_rmi_EchoServer_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.EchoServer._includes) */
#include "sidlx_common.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.EchoServer._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_EchoServer__ctor"

void
impl_sidlx_rmi_EchoServer__ctor(
  sidlx_rmi_EchoServer self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.EchoServer._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.EchoServer._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_EchoServer__dtor"

void
impl_sidlx_rmi_EchoServer__dtor(
  sidlx_rmi_EchoServer self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.EchoServer._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.EchoServer._dtor) */
}

/*
 * Method:  serviceRequest[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_EchoServer_serviceRequest"

void
impl_sidlx_rmi_EchoServer_serviceRequest(
  sidlx_rmi_EchoServer self, int32_t client_fd, sidl_BaseInterface* _ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.EchoServer.serviceRequest) */
  char * client_ip; 
  int client_port; 
  anysockaddr un;
  socklen_t len;
  char buff[MAXLINE];
  struct sidl_char__array * data=NULL; 
  int i, n;
  
  len=MAXSOCKADDR;
  s_getsockname(client_fd, (struct sockaddr  * ) un.data, &len, 
		_ex);
  if (*_ex) { printf("exception in s_getsockname\n"); }     SIDL_CHECK(*_ex);

  if (un.sa.sa_family == AF_INET ) { 
    struct sockaddr_in * cliaddr = (struct sockaddr_in *) &(un.sa);
    printf("EchoServer: connection to %s port %d\n",
	   inet_ntop(AF_INET, &cliaddr->sin_addr, buff, sizeof(buff)),
	   ntohs(cliaddr->sin_port )); 
  } else { 
    printf("connection using family %d\n", un.sa.sa_family );
  }

  printf("about to start\n");
  for ( i=0; i>-1; i++ ) { 
    printf("%d. ready to read...\n", i);
    n = s_readline(client_fd, MAXLINE, &data, _ex);
    if (*_ex) { printf("exception in s_readline\n"); }     SIDL_CHECK(*_ex);
    if (n == 0) {
      break;
    }
    s_writen(client_fd, data, _ex); 
    if (*_ex) { printf("exception in s_writen\n"); }     SIDL_CHECK(*_ex);
  }
 EXIT:
  printf("Exiting\n");
  if (data) { 
    sidl_char__array_deleteRef(data);
    data=NULL;
  }
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.EchoServer.serviceRequest) */
}
