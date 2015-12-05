/*
 * File:          sidlx_rmi_SimpleServer_Impl.c
 * Symbol:        sidlx.rmi.SimpleServer-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for sidlx.rmi.SimpleServer
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.SimpleServer" (version 0.1)
 * 
 * A multi-threaded base class for simple network servers.
 * 
 * This server takes the following flags:
 * 1: verbose output (to stdout)
 */

#include "sidlx_rmi_SimpleServer_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._includes) */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netdb.h>
#include <errno.h>
#include <sys/param.h>
#include "sidlx_common.h"
#include "sidlx_rmi_Common.h"
#include "sidl_rmi_NetworkException.h"
#include "sidl_String.h"
#include <pthread.h>

#define LISTENQ 1024
#define MAXLINE 1023

#define DEBUG 0

/* NOTE:  
 *       
 * the "g_" prefix is used to indicate global 
 * context that must be locked first
 */

static pthread_mutex_t g_poolLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_poolCond = PTHREAD_COND_INITIALIZER;

static sidl_bool g_shutdownPool = FALSE; /* flag to trigger shutdown */
static sidl_bool g_haveWork = FALSE; /* flag to indicate additional work 
				        available on g_server and g_socket */
static int g_nPool = 0;  /* number of worker threads in pool */
static int g_nBusy = 0;  /* number of threads busy */
static int g_maxPool = 1024; /* max number of threads in pool */

/* 
 * the following two are storage for serverFunc() to transfer 
 * information to threadFunc() 
 */
static sidlx_rmi_SimpleServer g_server = NULL;
static sidlx_rmi_Socket g_socket = NULL;

/**
 * This function is executed by child threads dealing with server connections.
 * It just runs the simpleServer serviceRequest, and cleans up after itself on exit.
 */
static void * threadFunc(void *arg) {
  sidlx_rmi_SimpleServer self = NULL;
  sidlx_rmi_Socket ac_sock = NULL;

  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface _ex2 = NULL;
  sidl_SIDLException e_x_p = NULL;

  while(1) {
    /* Try to grab some work */
    pthread_mutex_lock(&g_poolLock);

    while(! g_haveWork && !g_shutdownPool) {
      pthread_cond_wait(&g_poolCond,&g_poolLock);
    }
    if ( g_shutdownPool ) { 
      if (DEBUG) printf("Shutting down a thread %d\n",g_shutdownPool);
      g_nPool--;
      break;
    }

    g_nBusy++;                    /* Mark a busy thread */
    self = g_server;              /* grab work specified by serverFunc() */
    ac_sock = g_socket;           /* grab work specified by serverFunc() */
    g_haveWork = FALSE;           /* Mark work taken */
    pthread_cond_broadcast(&g_poolCond);
    pthread_mutex_unlock(&g_poolLock);

    sidlx_rmi_SimpleServer_serviceRequest( self, ac_sock, &_ex );
    SIDL_CHECK(_ex);

    sidlx_rmi_Socket_deleteRef(ac_sock,&_ex); SIDL_CHECK(_ex);

    /* Mark as not busy anymore */
    pthread_mutex_lock(&g_poolLock);
    g_nBusy--;
    pthread_mutex_unlock(&g_poolLock);

  } /* end while(1) */
  pthread_cond_broadcast(&g_poolCond);
  pthread_mutex_unlock(&g_poolLock);

  return NULL;
 EXIT:
  /* Assume that only coming here when busy and lock is unlocked */
  /* because that is the only place we have SIDL_CHECK() */

  pthread_mutex_lock(&g_poolLock);
  g_nBusy--;
  g_nPool--;
  pthread_mutex_unlock(&g_poolLock);

  /* Don't do SIDL_CHECK HERE!!! */
  sidlx_rmi_Socket_deleteRef(ac_sock,&_ex2); 
  e_x_p = sidl_SIDLException__cast(_ex,&_ex2);
  printf("Error in thread! %s\n", sidl_SIDLException_getNote(e_x_p,&_ex2));
  SIDL_CLEAR(_ex);
  return NULL;
}

/**
 * This function is executed by the thread that acts as server, accepting
 * connections and spawning children to deal with them.  It does not 
 * detatch itself because the main thread may need to join to it to prevent
 * premature exit.
 * It is also designed to work in a nonthreaded enviornment, but I haven't
 * tested that.
 */
static void * serverFunc(void *arg) {

  sidlx_rmi_SimpleServer self = (sidlx_rmi_SimpleServer)arg;
  sidlx_rmi_ServerSocket serverSocket = NULL;
  sidlx_rmi_Socket ac_sock = NULL;
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface _ex2 = NULL;
  sidl_SIDLException e_x_p = NULL;

  struct sidlx_rmi_SimpleServer__data *dptr=sidlx_rmi_SimpleServer__get_data(self);
  if(!dptr || !dptr->s_sock) {
    SIDL_THROW(_ex, sidl_rmi_NetworkException,"Simple Server not initialized");
  }

  /* Now that we know we'll be using this, addRef it */
  sidlx_rmi_SimpleServer_addRef(self, &_ex);
  serverSocket = dptr->s_sock;
  sidlx_rmi_ServerSocket_addRef(serverSocket, &_ex);

  while (1) { 
    pthread_t tid = 0;

    ac_sock = sidlx_rmi_ServerSocket_accept(serverSocket, &_ex); SIDL_CHECK(_ex);

    pthread_mutex_lock(&g_poolLock);
    if (g_shutdownPool) {
      pthread_mutex_unlock(&g_poolLock);
      if (DEBUG) printf("Have a connection, but we shouldn't service it!\n");
      goto EXIT;
    }
    if ( (g_nBusy == g_nPool) && (g_nPool < g_maxPool) ) {
      pthread_create(&tid,0,threadFunc,(void*)0);
      if(tid) {
        pthread_detach(tid);
        ++g_nPool;
      }
    }
    
    /* Make sure we don't have pending work */
    while(g_haveWork) {
      pthread_cond_wait(&g_poolCond,&g_poolLock);
    }
    /* Assign new work */
    g_haveWork = TRUE;
    g_server = self;
    g_socket = ac_sock;
    pthread_cond_broadcast(&g_poolCond);
    pthread_mutex_unlock(&g_poolLock);
  }
  pthread_mutex_unlock(&g_poolLock);
  return NULL;  /* See if we have any bored workers */
 EXIT:
  if (ac_sock) {
    sidlx_rmi_Socket_close(ac_sock,&_ex2);
    ac_sock = 0;
    SIDL_CLEAR(_ex2);
  }
  if (serverSocket) {
    sidlx_rmi_ServerSocket_deleteRef(serverSocket,&_ex2);
    SIDL_CLEAR(_ex2);
  }
  if (g_shutdownPool) {
    if (DEBUG) printf("Shutting down server thread cleanly!!!\n");
  } else {
    e_x_p = sidl_SIDLException__cast(_ex, &_ex2);
    printf("Error in thread! %s\n", sidl_SIDLException_getNote(e_x_p,&_ex2));
    printf("%s\n*******\n",sidl_SIDLException_getTrace(e_x_p,&_ex2));
  }
  SIDL_CLEAR(_ex);
  return NULL;  
}


/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer__ctor(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._ctor) */
  struct sidlx_rmi_SimpleServer__data *dptr;
  char name[MAXHOSTNAMELEN]; /*= sidl_String_alloc(MAXHOSTNAMELEN);*/

  dptr = malloc(sizeof(struct sidlx_rmi_SimpleServer__data));
  sidlx_rmi_SimpleServer__set_data(self, dptr);
  dptr->s_sock = sidlx_rmi_ServerSocket__create(_ex); SIDL_CHECK(*_ex);
  if(gethostname(name, MAXHOSTNAMELEN) == 0) { /*Returns 0 on success*/
    dptr->d_hostname =  sidlx_rmi_Common_getCanonicalName(name, _ex); SIDL_CHECK(*_ex);
  } else { 
    dptr->d_hostname = NULL;
  }
  dptr->d_port = -1;
  dptr->d_flags = 0;
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer__ctor2(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.SimpleServer._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer__dtor(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._dtor) */
  struct sidlx_rmi_SimpleServer__data * data = sidlx_rmi_SimpleServer__get_data( self );
  if (data) {
    if(data->s_sock) { sidlx_rmi_ServerSocket_deleteRef(data->s_sock, _ex); }
    free((void*) data);
  }
  sidlx_rmi_SimpleServer__set_data( self, NULL );
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._dtor) */
  }
}

/*
 * Set the maximum size of the client thread pool.
 * (default = 1024)
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_setMaxThreadPool"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer_setMaxThreadPool(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ int32_t max,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.setMaxThreadPool) */
  pthread_mutex_lock(&g_poolLock);
  g_maxPool = max;
  pthread_mutex_unlock(&g_poolLock);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.setMaxThreadPool) */
  }
}

/*
 * request a specific port number
 * returns true iff request is satisfied
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_requestPort"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_rmi_SimpleServer_requestPort(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.requestPort) */
  struct sidlx_rmi_SimpleServer__data *data=sidlx_rmi_SimpleServer__get_data(self);
  if(data) {
    sidlx_rmi_ServerSocket_init(data->s_sock, port, _ex); SIDL_CHECK(*_ex);
    data->d_port = port;
    return TRUE;
  }
  return FALSE;
 EXIT:
  SIDL_CLEAR(*_ex);
  return FALSE;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.requestPort) */
  }
}

/*
 * Request the minimum available port in 
 * a range.  Returns true
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_requestPortInRange"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_rmi_SimpleServer_requestPortInRange(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ int32_t minport,
  /* in */ int32_t maxport,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.requestPortInRange) */
  sidl_bool found=FALSE;
  int32_t i;
  for(i = minport; i<=maxport && !found; i++) { 
    found = sidlx_rmi_SimpleServer_requestPort(self, i, _ex); SIDL_CHECK(*_ex);
  }
  return found;
 EXIT:
  return FALSE;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.requestPortInRange) */
  }
}

/*
 * get the port that this Server is bound to
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_getPort"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_SimpleServer_getPort(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.getPort) */
  struct sidlx_rmi_SimpleServer__data *dptr=sidlx_rmi_SimpleServer__get_data(self);
  
  if(dptr) {
    return dptr->d_port;
  }
  return 0;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.getPort) */
  }
}

/*
 * get the network name of this computer
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_getServerName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimpleServer_getServerName(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.getServerName) */
  struct sidlx_rmi_SimpleServer__data *dptr = sidlx_rmi_SimpleServer__get_data(self);
  if ( dptr && dptr->d_hostname ) { 
    return sidl_String_strdup( dptr->d_hostname );
  } 
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.getServerName) */
  }
}

/*
 * get the full URL for exporting objects
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_getServerURL"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimpleServer_getServerURL(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ const char* objID,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.getServerURL) */
  /* TODO: Is there anything we can do here?*/
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.getServerURL) */
  }
}

/*
 * run the server (must have port specified first), if a threaded server,
 * returns the tid of the server thread for joining.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_run"

#ifdef __cplusplus
extern "C"
#endif
int64_t
impl_sidlx_rmi_SimpleServer_run(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.run) */
  pthread_t tid = 0;
  sidlx_rmi_SimpleServer_addRef(self, _ex); SIDL_CHECK(*_ex);
  pthread_create(&tid, NULL, (void * (*)(void *))&serverFunc, (void*)self);
  return (int64_t)tid;
 EXIT:
  return 0;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.run) */
  }
}

/*
 * cleanly shutdown the orb.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_shutdown"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer_shutdown(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.shutdown) */
 
  /* We close the socket so the server will not accept any more messages */
  pthread_mutex_lock(&g_poolLock);
  g_shutdownPool = TRUE;
  pthread_mutex_unlock(&g_poolLock);

  struct sidlx_rmi_SimpleServer__data *dptr=sidlx_rmi_SimpleServer__get_data(self);
  if (dptr && dptr->s_sock) {
    sidlx_rmi_ServerSocket_close(dptr->s_sock,_ex);
    sidlx_rmi_ServerSocket_deleteRef(dptr->s_sock, _ex);
    dptr->s_sock = 0;
  }
  if (*_ex) return;

  /* Wait until any current messages are done */
  pthread_mutex_lock(&g_poolLock);
  while(g_nBusy > 0) {
    pthread_cond_wait(&g_poolCond,&g_poolLock);
    if (DEBUG) printf("Trying to shutdown the orb... There are %d busy in pool of %d\n",
              g_nBusy,g_nPool);
  }
  
  g_shutdownPool = TRUE;
  pthread_cond_broadcast(&g_poolCond);

  while(g_nPool > 0) {
    pthread_cond_wait(&g_poolCond,&g_poolLock);
    /*if (DEBUG) printf("... %d remaining to shut down\n",g_nPool);*/
  }
  pthread_mutex_unlock(&g_poolLock);

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.shutdown) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_sidlx_rmi_SimpleServer_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_sidlx_rmi_SimpleServer_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_io_Serializable(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializable__connectI(url, ar, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializable__cast(bi, _ex);
}
struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_rmi_ServerInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_ServerInfo__connectI(url, ar, _ex);
}
struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_rmi_ServerInfo(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_rmi_ServerInfo__cast(bi, _ex);
}
struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimpleServer__connectI(url, ar, _ex);
}
struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_SimpleServer(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_SimpleServer__cast(bi, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_Socket__cast(bi, _ex);
}
