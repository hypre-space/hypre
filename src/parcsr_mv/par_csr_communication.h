/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_PAR_CSR_COMMUNICATION_HEADER
#define HYPRE_PAR_CSR_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef enum CommPkgJobType
{
   HYPRE_COMM_PKG_JOB_COMPLEX = 0,
   HYPRE_COMM_PKG_JOB_COMPLEX_TRANSPOSE,
   HYPRE_COMM_PKG_JOB_INT,
   HYPRE_COMM_PKG_JOB_INT_TRANSPOSE,
   HYPRE_COMM_PKG_JOB_BIGINT,
   HYPRE_COMM_PKG_JOB_BIGINT_TRANSPOSE,
   NUM_OF_COMM_PKG_JOB_TYPE
} CommPkgJobType;

static inline CommPkgJobType
hypre_ParCSRCommHandleGetJobType(HYPRE_Int job)
{
   CommPkgJobType job_type = HYPRE_COMM_PKG_JOB_COMPLEX;
   switch (job)
   {
      case  1:
         job_type = HYPRE_COMM_PKG_JOB_COMPLEX;
         break;
      case  2:
         job_type = HYPRE_COMM_PKG_JOB_COMPLEX_TRANSPOSE;
         break;
      case 11:
         job_type = HYPRE_COMM_PKG_JOB_INT;
         break;
      case 12:
         job_type = HYPRE_COMM_PKG_JOB_INT_TRANSPOSE;
         break;
      case 21:
         job_type = HYPRE_COMM_PKG_JOB_BIGINT;
         break;
      case 22:
         job_type = HYPRE_COMM_PKG_JOB_BIGINT_TRANSPOSE;
         break;
   }

   return job_type;
}

static inline HYPRE_Int
hypre_ParCSRCommHandleIsTransposeJob(HYPRE_Int job)
{
   HYPRE_Int trans = 0;

   switch (hypre_ParCSRCommHandleGetJobType(job))
   {
      case HYPRE_COMM_PKG_JOB_COMPLEX:
      case HYPRE_COMM_PKG_JOB_INT:
      case HYPRE_COMM_PKG_JOB_BIGINT:
      {
         trans = 0;
         break;
      }
      case HYPRE_COMM_PKG_JOB_COMPLEX_TRANSPOSE:
      case HYPRE_COMM_PKG_JOB_INT_TRANSPOSE:
      case HYPRE_COMM_PKG_JOB_BIGINT_TRANSPOSE:
      {
         trans = 1;
         break;
      }
      default:
         break;
   }
   return trans;
}

static inline hypre_MPI_Datatype
hypre_ParCSRCommHandleGetMPIDataType(HYPRE_Int job)
{
   hypre_MPI_Datatype dtype = HYPRE_MPI_COMPLEX;

   switch (hypre_ParCSRCommHandleGetJobType(job))
   {
      case HYPRE_COMM_PKG_JOB_COMPLEX:
      case HYPRE_COMM_PKG_JOB_COMPLEX_TRANSPOSE:
         dtype = HYPRE_MPI_COMPLEX;
         break;
      case HYPRE_COMM_PKG_JOB_INT:
      case HYPRE_COMM_PKG_JOB_INT_TRANSPOSE:
         dtype = HYPRE_MPI_INT;
         break;
      case HYPRE_COMM_PKG_JOB_BIGINT:
      case HYPRE_COMM_PKG_JOB_BIGINT_TRANSPOSE:
         dtype = HYPRE_MPI_BIG_INT;
         break;
      default:
         break;
   }

   return dtype;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommHandle, hypre_ParCSRPersistentCommHandle
 *--------------------------------------------------------------------------*/
struct _hypre_ParCSRCommPkg;

typedef struct
{
   struct _hypre_ParCSRCommPkg *comm_pkg;
   HYPRE_Int                    persistent;
   void                        *send_data;
   void                        *recv_data;
   HYPRE_MemoryLocation         send_location;
   HYPRE_MemoryLocation         recv_location;
   HYPRE_Int                    num_requests;
   hypre_MPI_Request           *requests;
   hypre_MPICommWrapper        *comm;
} hypre_ParCSRCommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommHandle
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRCommHandleCommPkg(comm_handle)                (comm_handle -> comm_pkg)
#define hypre_ParCSRCommHandlePersistent(comm_handle)             (comm_handle -> persistent)
#define hypre_ParCSRCommHandleSendData(comm_handle)               (comm_handle -> send_data)
#define hypre_ParCSRCommHandleRecvData(comm_handle)               (comm_handle -> recv_data)
#define hypre_ParCSRCommHandleSendLocation(comm_handle)           (comm_handle -> send_location)
#define hypre_ParCSRCommHandleRecvLocation(comm_handle)           (comm_handle -> recv_location)
#define hypre_ParCSRCommHandleNumRequests(comm_handle)            (comm_handle -> num_requests)
#define hypre_ParCSRCommHandleRequests(comm_handle)               (comm_handle -> requests)
#define hypre_ParCSRCommHandleRequest(comm_handle, i)             (comm_handle -> requests[i])
#define hypre_ParCSRCommHandleComm(comm_handle)                   (comm_handle -> comm)

typedef hypre_ParCSRCommHandle hypre_ParCSRPersistentCommHandle;

typedef struct _hypre_ParCSRCommPkg
{
   MPI_Comm                          comm;
   HYPRE_Int                         num_components;
   HYPRE_Int                         num_sends;
   HYPRE_Int                        *send_procs;
   HYPRE_Int                        *send_map_starts;
   HYPRE_Int                        *send_map_elmts;
   HYPRE_Int                        *device_send_map_elmts;
   HYPRE_Int                         num_recvs;
   HYPRE_Int                        *recv_procs;
   HYPRE_Int                        *recv_vec_starts;
   /* remote communication information */
   hypre_MPI_Datatype               *send_mpi_types;
   hypre_MPI_Datatype               *recv_mpi_types;
#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandle *persistent_comm_handles[NUM_OF_COMM_PKG_JOB_TYPE];
#endif
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   /* temporary memory for matvec. cudaMalloc is expensive. alloc once and reuse */
   HYPRE_Complex                    *tmp_data;
   HYPRE_Complex                    *buf_data;
   hypre_CSRMatrix                  *matrix_E; /* for matvecT */
#endif
} hypre_ParCSRCommPkg;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommPkg
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRCommPkgComm(comm_pkg)                        (comm_pkg -> comm)
#define hypre_ParCSRCommPkgNumComponents(comm_pkg)               (comm_pkg -> num_components)
#define hypre_ParCSRCommPkgNumSends(comm_pkg)                    (comm_pkg -> num_sends)
#define hypre_ParCSRCommPkgSendProcs(comm_pkg)                   (comm_pkg -> send_procs)
#define hypre_ParCSRCommPkgSendProc(comm_pkg, i)                 (comm_pkg -> send_procs[i])
#define hypre_ParCSRCommPkgSendMapStarts(comm_pkg)               (comm_pkg -> send_map_starts)
#define hypre_ParCSRCommPkgSendMapStart(comm_pkg,i)              (comm_pkg -> send_map_starts[i])
#define hypre_ParCSRCommPkgSendMapElmts(comm_pkg)                (comm_pkg -> send_map_elmts)
#define hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg)          (comm_pkg -> device_send_map_elmts)
#define hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)               (comm_pkg -> send_map_elmts[i])
#define hypre_ParCSRCommPkgDeviceSendMapElmt(comm_pkg,i)         (comm_pkg -> device_send_map_elmts[i])
#define hypre_ParCSRCommPkgNumRecvs(comm_pkg)                    (comm_pkg -> num_recvs)
#define hypre_ParCSRCommPkgRecvProcs(comm_pkg)                   (comm_pkg -> recv_procs)
#define hypre_ParCSRCommPkgRecvProc(comm_pkg, i)                 (comm_pkg -> recv_procs[i])
#define hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)               (comm_pkg -> recv_vec_starts)
#define hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i)              (comm_pkg -> recv_vec_starts[i])
#define hypre_ParCSRCommPkgSendMPITypes(comm_pkg)                (comm_pkg -> send_mpi_types)
#define hypre_ParCSRCommPkgSendMPIType(comm_pkg,i)               (comm_pkg -> send_mpi_types[i])
#define hypre_ParCSRCommPkgRecvMPITypes(comm_pkg)                (comm_pkg -> recv_mpi_types)
#define hypre_ParCSRCommPkgRecvMPIType(comm_pkg,i)               (comm_pkg -> recv_mpi_types[i])
#define hypre_ParCSRCommPkgPersistentCommHandles(comm_pkg)       (comm_pkg -> persistent_comm_handles)
#define hypre_ParCSRCommPkgPersistentCommHandle(comm_pkg,i)      (comm_pkg -> persistent_comm_handles[i])

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
#define hypre_ParCSRCommPkgTmpData(comm_pkg)                     ((comm_pkg) -> tmp_data)
#define hypre_ParCSRCommPkgBufData(comm_pkg)                     ((comm_pkg) -> buf_data)
#define hypre_ParCSRCommPkgMatrixE(comm_pkg)                     ((comm_pkg) -> matrix_E)
#endif

static inline HYPRE_MAYBE_UNUSED_FUNC void
hypre_ParCSRCommPkgCopySendMapElmtsToDevice(hypre_ParCSRCommPkg *comm_pkg)
{
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) == NULL)
   {
      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) =
         hypre_TAlloc(HYPRE_Int,
                      hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                      HYPRE_MEMORY_DEVICE);

      hypre_TMemcpy(hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                    hypre_ParCSRCommPkgSendMapElmts(comm_pkg),
                    HYPRE_Int,
                    hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                    HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);
   }
#else
   HYPRE_UNUSED_VAR(comm_pkg);
#endif
}

#endif /* HYPRE_PAR_CSR_COMMUNICATION_HEADER */
