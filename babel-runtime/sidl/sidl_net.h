
  struct sidl_net_Connection { 
    char * protocol;
    char * server_name;
    int port;
    char * path;
    char * session_id;
  };

  struct sidl_net_Connection * sidl_net_Connection__create( /*in*/ const char* url );

  void sidl_net_Connection__destroy( /*in*/ struct sidl_net_Connection* conn );
