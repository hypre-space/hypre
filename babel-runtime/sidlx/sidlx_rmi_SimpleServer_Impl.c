/*
 * File:          sidlx_rmi_SimpleServer_Impl.c
 * Symbol:        sidlx.rmi.SimpleServer-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.rmi.SimpleServer
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
 * Symbol "sidlx.rmi.SimpleServer" (version 0.1)
 * 
 * A multi-threaded base class for simple network servers.
 */

#include "sidlx_rmi_SimpleServer_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._includes) */
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include "sidlx_common.h"
#define LISTENQ 1024
#define MAXLINE 1023
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer__ctor"

void
impl_sidlx_rmi_SimpleServer__ctor(
  sidlx_rmi_SimpleServer self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._ctor) */
  int i;

  sidlX_MALLOC_SET_DATA( sidlx_rmi_SimpleServer );

  bzero( &(data->d_serv_addr), sizeof( struct sockaddr_in ));

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer__dtor"

void
impl_sidlx_rmi_SimpleServer__dtor(
  sidlx_rmi_SimpleServer self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._dtor) */
  struct sidlx_rmi_SimpleServer__data * data = sidlx_rmi_SimpleServer__get_data( self );
  if (data) { 
    free((void*) data);
  }
  sidlx_rmi_SimpleServer__set_data( self, NULL );
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._dtor) */
}

/*
 * set which port number to bind to
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_setPort"

void
impl_sidlx_rmi_SimpleServer_setPort(
  sidlx_rmi_SimpleServer self, int32_t port)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.setPort) */
  struct sidlx_rmi_SimpleServer__data *data=sidlx_rmi_SimpleServer__get_data(self);
  data->d_serv_addr.sin_family = AF_INET;
  data->d_serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  data->d_serv_addr.sin_port = htons( port );
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.setPort) */
}

/*
 * run the server (must have port specified first)
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_run"

void
impl_sidlx_rmi_SimpleServer_run(
  sidlx_rmi_SimpleServer self, sidl_BaseInterface* _ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.run) */
  int i;
  int len;
  int connection_socket;
  int pid;
  struct sockaddr_in cliaddr;
  struct sidlx_rmi_SimpleServer__data *data;
  char buff[MAXLINE];

  data=sidlx_rmi_SimpleServer__get_data(self);

  data->d_listen_socket = 
    s_socket( AF_INET, SOCK_STREAM, 0, _ex); SIDL_CHECK(*_ex);

  s_bind( data->d_listen_socket, (struct sockaddr*) &(data->d_serv_addr), 
	  sizeof (data->d_serv_addr), _ex); SIDL_CHECK(*_ex);
  
  s_listen( data->d_listen_socket, LISTENQ, _ex); SIDL_CHECK(*_ex);

  for(;;) { 
    len = sizeof(cliaddr);
    connection_socket = s_accept( data->d_listen_socket, 
				  (struct sockaddr*) &cliaddr, &len, _ex);
    if ( errno == EINTR ) { 
      SIDL_CLEAR(*_ex);
      continue;
    } else { 
      SIDL_CHECK(*_ex);
    }

#ifdef GARY_K
     pid = s_fork(_ex); SIDL_CHECK(*_ex);
       if ( pid == 0 ) { 
      /* child closes listening socket */
      s_close( data->d_listen_socket, _ex); SIDL_CHECK(*_ex);
#endif
      /* process request */
      printf("SimpleServer: connection from %s port %d\n",
	     inet_ntop(AF_INET, &cliaddr.sin_addr, buff, sizeof(buff)),
	     ntohs(cliaddr.sin_port ));
      sidlx_rmi_SimpleServer_serviceRequest( self, connection_socket,_ex );
      /*s_close(connection_socket, _ex); SIDL_CHECK(*_ex);*/
      exit(0);
#ifdef GARY_K
      } 
	s_close(connection_socket, _ex); SIDL_CHECK(*_ex);
#endif
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.run) */
}
