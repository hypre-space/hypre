/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParVector Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_CreateParVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_createparvector)( int *comm,
                                        int *global_size,
                                        int *partitioning
                                        int *ierr )
{
   *ierr = (int) ( HYPRE_CreateParVector( (MPI_Comm) *comm,
                                          (int)      *global_size,
                                          (int *)    partitioning  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DestroyParVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_destroyparvector)( long int *vector,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_DestroyParVector( (HYPRE_ParCSRMatrix) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeParVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_initializeparvector)( long int *vector,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_InitializeParVector( (HYPRE_ParCSRMatrix) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintParVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_printparvector)( long int *vector,
                                       char     *file_name,
                                       int      *ierr       )
{
   *ierr = (int) ( HYPRE_PrintParVector ( (HYPRE_ParCSRMatrix) *vector,
                                          (char *)             file_name ) );
}

