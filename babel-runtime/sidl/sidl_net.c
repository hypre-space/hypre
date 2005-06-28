#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sidl_net.h"


struct sidl_net_Connection* sidl_net_Connection__create( /*in*/ const char* url ) { 
  struct sidl_net_Connection *conn = malloc(sizeof(struct sidl_net_Connection));
  int end = strlen(url), 
      i = 0, start = i,
      size;

  /* extract protocol name */
  while ((i < end) && (url[i] != ':')) {
    i++;
  }

  if ((i == start) || (i == end)) {
    printf("ERROR: invalid URL format\n");
    exit(2);
  }

  /* printf("i = %i\n", i); */
  size = i - start;
  conn->protocol = malloc(size + 1);
  memset(conn->protocol, 0 , size + 1);
  strncpy(conn->protocol, url + start, size);

  /* printf("protocol:    %s\n", conn->protocol); */

  /* extract server name   */
  i += 3; /* skip the double slashes */
  start = i;
  while ((i < end) && (url[i] != ':')) {
    i++;
  }

  if ((i == start) || (i == end)) {
    printf("ERROR: invalid URL format\n");
    exit(2);
  }

  size = i - start;
  conn->server_name = malloc(size + 1);
  memset(conn->server_name, 0 , size + 1);
  strncpy(conn->server_name, url + start, size);

  /* printf("server_name: %s\n", conn->server_name); */

  /* extract port number */

  i++; /* skip the colon */
  start = i;
  while ((i < end) && (url[i] != '/')) {
    if ((url[i] < 48) || (url[i] > 57))
      printf("ERROR: invalid URL format\n");
    i++;
  }

  conn->port = atoi(url + start);

  /* printf("port:        %i\n", conn->port); */

  /* extract path */

  i++; /* skip the single slash */
  start = i;

  if (i < end) {
    size = end - start;
    conn->path = malloc(size + 1);
    memset(conn->path, 0 , size + 1);
    strcpy(conn->path, url + start);
    /* printf("path:        %s\n", conn->path); */
  }

  return conn;
}

void sidl_net_Connection__destroy( /*in*/ struct sidl_net_Connection* conn ) { 
  free(conn->protocol);
  free(conn->server_name);
  free(conn->path);
  free(conn->session_id);
  free(conn);
}
