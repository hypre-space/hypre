/*****************************************************************************
 *
 * HYPRE_par_laplace Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * GenerateLaplacian
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_generatelaplacian, HYPRE_GENERATELAPLACIAN)
                                  ( int      *comm,
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
                                    long int *matrix,
                                    int      *ierr   )

{
   *matrix = (long int) ( GenerateLaplacian( (MPI_Comm) *comm,
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

