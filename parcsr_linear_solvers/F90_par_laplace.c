/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * par_laplace Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * hypre_GenerateLaplacian
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_generatelaplacian)( int      *comm,
                                          int      *nx,
                                          int      *ny,
                                          int      *nz,
                                          int      *P,
                                          int      *Q,
                                          int      *R,
                                          int      *p,
                                          int      *q,
                                          int      *r,
                                          double   *value,
                                          long int *matrix
                                          int      *ierr   )

{
   *matrix = (long int) ( hypre_GenerateLaplacian( (MPI_Comm) *comm,
                                                   (int)      *nx,
                                                   (int)      *ny,
                                                   (int)      *nz,
                                                   (int)      *P,
                                                   (int)      *Q,
                                                   (int)      *R,
                                                   (int)      *p,
                                                   (int)      *q,
                                                   (int)      *r,
                                                   (double *)  value ) );

   *ierr = 0;
}

