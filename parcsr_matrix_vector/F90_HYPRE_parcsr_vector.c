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
 * HYPRE_NewParVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_newparvector)( int      *comm,
                                     int      *global_size,
                                     long int *partitioning,
                                     long int *vector,
                                     int      *ierr          )
{
   *vector = (long int)
             ( HYPRE_CreateParVector( (MPI_Comm) *comm,
                                      (int)      *global_size,
                                      (int *)    *partitioning ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_DestroyParVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_destroyparvector)( long int *vector,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_DestroyParVector( (HYPRE_ParVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeParVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_initializeparvector)( long int *vector,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_InitializeParVector( (HYPRE_ParVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintParVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_printparvector)( long int *vector,
                                       char     *file_name,
                                       int      *ierr       )
{
   *ierr = (int) ( HYPRE_PrintParVector ( (HYPRE_ParVector) *vector,
                                          (char *)           file_name ) );
}

