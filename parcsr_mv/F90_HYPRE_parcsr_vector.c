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
 * HYPRE_ParVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcreate)( int      *comm,
                                     int      *global_size,
                                     long int *partitioning,
                                     long int *vector,
                                     int      *ierr          )
{
   *ierr = (long int)
             ( HYPRE_ParVectorCreate( (MPI_Comm) *comm,
                                      (int)      *global_size,
                                      (int *)    *partitioning,
                                      (HYPRE_ParVector *)  vector ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectordestroy)( long int *vector,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorDestroy( (HYPRE_ParVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorinitialize)( long int *vector,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorInitialize( (HYPRE_ParVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectorread)( int      *comm,
                                      long int *vector,
                                      char     *file_name,
                                      int      *ierr       )
{
   *ierr = (int) ( HYPRE_ParVectorRead( (MPI_Comm) *comm,
                                        (char *)    file_name,
                                        (HYPRE_ParVector *) vector ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorPrint
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectorprint)( long int *vector,
                                       char     *file_name,
                                       int      *ierr       )
{
   *ierr = (int) ( HYPRE_ParVectorPrint ( (HYPRE_ParVector) *vector,
                                          (char *)           file_name ) );
}

