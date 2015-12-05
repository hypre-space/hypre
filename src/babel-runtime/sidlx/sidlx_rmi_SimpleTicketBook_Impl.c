/*
 * File:          sidlx_rmi_SimpleTicketBook_Impl.c
 * Symbol:        sidlx.rmi.SimpleTicketBook-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for sidlx.rmi.SimpleTicketBook
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.SimpleTicketBook" (version 0.1)
 */

#include "sidlx_rmi_SimpleTicketBook_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook._includes) */
#include <stddef.h>
#include <stdlib.h>
#include "sidl_Exception.h"
#include <unistd.h>

static struct ticket_list* newList(void) {

  struct ticket_list* head = NULL;

  head = malloc( sizeof(struct ticket_list) );

  head->next = NULL;
  head->id = -1;
  head->ticket = NULL;
  return head;

}

/* Return TRUE if the list is empty */
static int isEmpty(struct ticket_list* head) {
  return head->next == NULL;
}

/*
 * Return TRUE if this node is the last node in the list.
 */
static int isLast(struct ticket_list* node) {
  return node->next == NULL;
}

/* Return the node corrosponding to the inputed id. NULL if not found */
static struct ticket_list* find(struct ticket_list* head, int32_t id) {

  struct ticket_list* node = NULL;

  node = head->next;
  while(node != NULL && node->id != id) {
    node = node->next;
  }
  return node;
}

/*
 * If id is not found, then Next field of returned
 * value is NULL
 * This implementation assumes the use of
 * a header node.
 */
static struct ticket_list* findPrevious(struct ticket_list* head, int32_t id) {

  struct ticket_list* node = NULL;

  node = head;
  while (node->next != NULL && node->next->id != id) {
    node = node->next;
  }

 return node;

}

/*
 * Finds the node with id number id, and deletes it.  
 * (If it exists)
 */
static void delete(struct ticket_list* head, 
		   int32_t id,
		   sidl_BaseInterface *_ex) {

  struct ticket_list* node = NULL;
  struct ticket_list* tmp = NULL;

  node = findPrevious(head, id);

  if(!node) { return;}

  /* Assumption use of a header node */
  if (node->next != NULL) {

    /* id was found by findPrevious.  Delete the node. */
    tmp = node->next;

    /* Bypass deleted node */
    node->next = tmp->next;
    if(tmp->ticket) {sidl_rmi_Ticket_deleteRef(tmp->ticket, _ex);}
    free(tmp);

  }

}

/*
 * Insert at the end of the list. 
 * This assumes that node has a valid id number and ticket.
 * This implementation assumes the use of a header node.
 */
static void insert(struct ticket_list* head, struct ticket_list* node) {

  struct ticket_list* tmp = NULL;

  tmp = head;
  while(tmp->next != NULL) {
    tmp = tmp->next;
  }
  tmp->next = node;
  node->next = NULL;

}


/* Correct DeleteList algorithm 
 * FREES EVERYHTING INCLUDING THE HEADER NODE
 */
static void deleteList(struct ticket_list* head, 
		       sidl_BaseInterface *_ex) {
  
  struct ticket_list* node = NULL;
  struct ticket_list* tmp = NULL;
  
  node = head;
  while(node != NULL) {
    tmp = node->next;
    if(node->ticket) {sidl_rmi_Ticket_deleteRef(node->ticket, _ex);}
    free(node);
    node = tmp;
  }

}


/* creates and initializes a single node */ 
static struct ticket_list* nodeCreate(void) {  
  struct ticket_list* node = NULL; 
  
  if ( ( node = (struct ticket_list*)  
 	 malloc( sizeof( struct ticket_list))) == NULL ) {  
    return NULL; 
  } else {  
    node->ticket = NULL; 
    node->id = -1; 
    node->next = NULL; 
    return node; 
  } 
} 




/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleTicketBook__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook._load) */
  /* Insert-Code-Here {sidlx.rmi.SimpleTicketBook._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleTicketBook__ctor(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook._ctor) */
  struct sidlx_rmi_SimpleTicketBook__data* data = (struct sidlx_rmi_SimpleTicketBook__data*) 
    malloc( sizeof( struct sidlx_rmi_SimpleTicketBook__data*));
  data->head = newList();
  sidlx_rmi_SimpleTicketBook__set_data(self, data);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleTicketBook__ctor2(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.SimpleTicketBook._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleTicketBook__dtor(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook._dtor) */
  struct sidlx_rmi_SimpleTicketBook__data* data = sidlx_rmi_SimpleTicketBook__get_data(self);
  if ( data != NULL ) { 
    deleteList(data->head,_ex); SIDL_CHECK(*_ex);
    free( data );
    data = NULL;
  }
  sidlx_rmi_SimpleTicketBook__set_data(self, NULL);
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook._dtor) */
  }
}

/*
 *  insert a ticket with a user-specified ID 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook_insertWithID"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleTicketBook_insertWithID(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* in */ sidl_rmi_Ticket t,
  /* in */ int32_t id,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook.insertWithID) */
  struct sidlx_rmi_SimpleTicketBook__data* data = sidlx_rmi_SimpleTicketBook__get_data(self);
  struct ticket_list* node = nodeCreate();
  sidl_rmi_Ticket_addRef(t, _ex); SIDL_CHECK(*_ex);
  node->ticket = t;
  node->id = id;
  insert(data->head, node );
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook.insertWithID) */
  }
}

/*
 *  insert a ticket and issue a unique ID 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook_insert"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_SimpleTicketBook_insert(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* in */ sidl_rmi_Ticket t,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook.insert) */

  struct sidlx_rmi_SimpleTicketBook__data* data = sidlx_rmi_SimpleTicketBook__get_data(self);
  struct ticket_list* node = data->head->next;
  int32_t id = 0;
  /* find the max id */
  for ( ; node != NULL; node = node->next ) { 
    id = (id < node->id ) ? node->id : id;
  }
  id++; /* now id is unique */
  node = nodeCreate();
  sidl_rmi_Ticket_addRef(t, _ex); SIDL_CHECK(*_ex);
  node->ticket = t;
  node->id = id;
  insert( data->head, node );
  return id;
 EXIT:
  return -1;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook.insert) */
  }
}

/*
 *  remove a ready ticket from the TicketBook
 * returns 0 (and null) on an empty TicketBook
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook_removeReady"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_SimpleTicketBook_removeReady(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* out */ sidl_rmi_Ticket* t,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook.removeReady) */
  struct sidlx_rmi_SimpleTicketBook__data* data = sidlx_rmi_SimpleTicketBook__get_data(self);
  int32_t id = -1;
  *t = NULL;

  for (;;) { 
    struct ticket_list * cur = data->head->next;
    if ( cur == NULL ) { return id; }
    
    for ( ; cur != NULL; cur = cur->next) { 
      if ( sidl_rmi_Ticket_test(cur->ticket, _ex) ) { 
	*t = cur->ticket;
	sidl_rmi_Ticket_addRef(*t,_ex); SIDL_CHECK(*_ex);
	id = cur->id;
	delete(data->head,id, _ex); SIDL_CHECK(*_ex);
	return id;
      }
    }
    sleep(5);
  }
 EXIT:
  return id;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook.removeReady) */
  }
}

/*
 * immediate, returns the number of Tickets in the book.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook_isEmpty"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_rmi_SimpleTicketBook_isEmpty(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook.isEmpty) */
  struct sidlx_rmi_SimpleTicketBook__data* data = sidlx_rmi_SimpleTicketBook__get_data(self);
  return isEmpty(data->head);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook.isEmpty) */
  }
}

/*
 *  blocks until the Response is recieved 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook_block"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleTicketBook_block(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook.block) */
  struct sidlx_rmi_SimpleTicketBook__data* data = sidlx_rmi_SimpleTicketBook__get_data(self);

  struct ticket_list * cur = data->head->next;
  for ( ; cur != NULL; cur = cur->next) { 
    sidl_rmi_Ticket_block(cur->ticket, _ex); SIDL_CHECK(*_ex);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook.block) */
  }
}

/*
 *  
 * returns immediately: true iff the Response is already
 * received 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook_test"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_rmi_SimpleTicketBook_test(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook.test) */
  struct sidlx_rmi_SimpleTicketBook__data* data = sidlx_rmi_SimpleTicketBook__get_data(self);
  struct ticket_list * cur = data->head->next;
  sidl_bool result = FALSE;
  if ( cur == NULL ) { return FALSE; }
  
  for ( ; cur != NULL; cur = cur->next) { 
    result = sidl_rmi_Ticket_test(cur->ticket, _ex); SIDL_CHECK(*_ex);
    if ( result == TRUE ) { 
      return TRUE;
    }
  }
 EXIT:
  return FALSE;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook.test) */
  }
}

/*
 *  creates an empty container specialized for Tickets 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook_createEmptyTicketBook"

#ifdef __cplusplus
extern "C"
#endif
sidl_rmi_TicketBook
impl_sidlx_rmi_SimpleTicketBook_createEmptyTicketBook(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook.createEmptyTicketBook) */
  sidlx_rmi_SimpleTicketBook book = sidlx_rmi_SimpleTicketBook__create(_ex); SIDL_CHECK(*_ex);
  sidl_rmi_TicketBook tb = sidl_rmi_TicketBook__cast(book, _ex); SIDL_CHECK(*_ex);
  sidlx_rmi_SimpleTicketBook_deleteRef(book,_ex); SIDL_CHECK(*_ex);
  return tb;
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook.createEmptyTicketBook) */
  }
}

/*
 *  returns immediately: returns Response or null 
 * (NOTE: needed for implementors of communication
 * libraries, not expected for general use).
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleTicketBook_getResponse"

#ifdef __cplusplus
extern "C"
#endif
sidl_rmi_Response
impl_sidlx_rmi_SimpleTicketBook_getResponse(
  /* in */ sidlx_rmi_SimpleTicketBook self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleTicketBook.getResponse) */
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleTicketBook.getResponse) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleTicketBook_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleTicketBook_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleTicketBook_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleTicketBook_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleTicketBook_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleTicketBook_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleTicketBook_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleTicketBook_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_SimpleTicketBook_fconnect_sidl_rmi_Response(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_Response__connectI(url, ar, _ex);
}
struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_SimpleTicketBook_fcast_sidl_rmi_Response(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_rmi_Response__cast(bi, _ex);
}
struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_SimpleTicketBook_fconnect_sidl_rmi_Ticket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_Ticket__connectI(url, ar, _ex);
}
struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_SimpleTicketBook_fcast_sidl_rmi_Ticket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_rmi_Ticket__cast(bi, _ex);
}
struct sidl_rmi_TicketBook__object* 
  impl_sidlx_rmi_SimpleTicketBook_fconnect_sidl_rmi_TicketBook(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_TicketBook__connectI(url, ar, _ex);
}
struct sidl_rmi_TicketBook__object* 
  impl_sidlx_rmi_SimpleTicketBook_fcast_sidl_rmi_TicketBook(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_rmi_TicketBook__cast(bi, _ex);
}
struct sidlx_rmi_SimpleTicketBook__object* 
  impl_sidlx_rmi_SimpleTicketBook_fconnect_sidlx_rmi_SimpleTicketBook(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimpleTicketBook__connectI(url, ar, _ex);
}
struct sidlx_rmi_SimpleTicketBook__object* 
  impl_sidlx_rmi_SimpleTicketBook_fcast_sidlx_rmi_SimpleTicketBook(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_SimpleTicketBook__cast(bi, _ex);
}
