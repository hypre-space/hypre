/*--------------------------------------------------------------------------
 * hypre_VectorCommPkg:
 *   Structure containing information for doing communications for parallel 
 *   vectors
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm               comm;

   int                   *vec_starts;

   /* remote communication information */
   MPI_Datatype          *vector_mpi_types;

} hypre_VectorCommPkg;

/*--------------------------------------------------------------------------
 * hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm               comm;

   int                    num_sends;
   int                   *send_procs;
   int			 *send_map_starts;
   int			 *send_map_elmts;

   int                    num_recvs;
   int                   *recv_procs;
   int                   *recv_vec_starts;

   /* remote communication information */
   MPI_Datatype          *send_mpi_types;
   MPI_Datatype          *recv_mpi_types;

} hypre_CommPkg;

/*--------------------------------------------------------------------------
 * hypre_CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_CommPkg  *comm_pkg;
   double         *send_data;
   double         *recv_data;

   int             num_requests;
   MPI_Request    *requests;

} hypre_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_VectorCommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_VectorCommPkgComm(comm_pkg)           (comm_pkg -> comm)
#define hypre_VectorCommPkgVecStarts(comm_pkg)      (comm_pkg -> vec_starts)
#define hypre_VectorCommPkgVecStart(comm_pkg,i)     (comm_pkg -> vec_starts[i])
#define hypre_VectorCommPkgVectorMPITypes(comm_pkg) (comm_pkg -> vector_mpi_types)
#define hypre_VectorCommPkgVectorMPIType(comm_pkg,i)(comm_pkg -> vector_mpi_types[i])
                                               
/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_CommPkgComm(comm_pkg)          (comm_pkg -> comm)
                                               
#define hypre_CommPkgNumSends(comm_pkg)      (comm_pkg -> num_sends)
#define hypre_CommPkgSendProcs(comm_pkg)     (comm_pkg -> send_procs)
#define hypre_CommPkgSendProc(comm_pkg, i)   (comm_pkg -> send_procs[i])
#define hypre_CommPkgSendMapStarts(comm_pkg) (comm_pkg -> send_map_starts)
#define hypre_CommPkgSendMapStart(comm_pkg,i)(comm_pkg -> send_map_starts[i])
#define hypre_CommPkgSendMapElmts(comm_pkg)  (comm_pkg -> send_map_elmts)
#define hypre_CommPkgSendMapElmt(comm_pkg,i) (comm_pkg -> send_map_elmts[i])

#define hypre_CommPkgNumRecvs(comm_pkg)      (comm_pkg -> num_recvs)
#define hypre_CommPkgRecvProcs(comm_pkg)     (comm_pkg -> recv_procs)
#define hypre_CommPkgRecvProc(comm_pkg, i)   (comm_pkg -> recv_procs[i])
#define hypre_CommPkgRecvVecStarts(comm_pkg) (comm_pkg -> recv_vec_starts)
#define hypre_CommPkgRecvVecStart(comm_pkg,i)(comm_pkg -> recv_vec_starts[i])

#define hypre_CommPkgSendMPITypes(comm_pkg)  (comm_pkg -> send_mpi_types)
#define hypre_CommPkgSendMPIType(comm_pkg,i) (comm_pkg -> send_mpi_types[i])

#define hypre_CommPkgRecvMPITypes(comm_pkg)  (comm_pkg -> recv_mpi_types)
#define hypre_CommPkgRecvMPIType(comm_pkg,i) (comm_pkg -> recv_mpi_types[i])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_CommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_CommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_CommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_CommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_CommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_CommHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

