/*
 * File:          sidlx_rmi_SimpleOrb_Impl.c
 * Symbol:        sidlx.rmi.SimpleOrb-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.rmi.SimpleOrb
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
 * Symbol "sidlx.rmi.SimpleOrb" (version 0.1)
 * 
 * An incomplete crack at a an orb
 */

#include "sidlx_rmi_SimpleOrb_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._includes) */
#include "sidlx_common.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb__ctor"

void
impl_sidlx_rmi_SimpleOrb__ctor(
  sidlx_rmi_SimpleOrb self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb__dtor"

void
impl_sidlx_rmi_SimpleOrb__dtor(
  sidlx_rmi_SimpleOrb self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._dtor) */
}

/*
 * Method:  serviceRequest[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb_serviceRequest"

void
impl_sidlx_rmi_SimpleOrb_serviceRequest(
  sidlx_rmi_SimpleOrb self, int32_t client_fd, sidl_BaseInterface* _ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb.serviceRequest) */
  char * client_ip; 
  int client_port; 
  anysockaddr un;
  socklen_t len;
  char buff[MAXLINE];
  
  len=MAXSOCKADDR;
  s_getsockname(client_fd, (struct sockaddr  * ) un.data, &len, 
		&_ex); SIDL_CHECK(*_ex);

  if (un.sa.sa_family == AF_INET ) { 
    struct sockaddr_in * cliaddr = (struct sockaddr_in *) &(un.sa);
    printf("SimpleOrb: connection from %s port %d\n",
	   inet_ntop(AF_INET, &cliaddr->sin_addr, buff, sizeof(buff)),
	   ntohs(cliaddr->sin_port )); 
  } else { 
    printf("connection using family %d\n", un.sa.sa_family );
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb.serviceRequest) */
}
