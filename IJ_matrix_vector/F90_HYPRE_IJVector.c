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
 * HYPRE_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./IJ_matrix_vector.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewIJVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_newijvector)( int      *comm,
                                    long int *vector,
                                    int      *global_n,
                                    int      *ierr      )
{
   *ierr = (int) ( HYPRE_NewIJVector( (MPI_Comm)         *comm,
                                      (HYPRE_IJVector *)  vector,
                                      (int)              *global_n  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeIJVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_freeijvector)( long int *vector,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_FreeIJVector( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeIJVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_initializeijvector)( long int *vector,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_InitializeIJVector( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleIJVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_assembleijvector)( long int *vector,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_AssembleIJVector ( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributeIJVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_distributeijvector)( long int *vector,
                                           int      *row_starts,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_DistributeIJVector( (HYPRE_IJVector) *matrix,
                                             (int *)           row_starts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalStorageT
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setijvectorlocalstoragety)( long int *vector,
                                                  int      *type,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_SetIJMatrixLocalStorageType( (HYPRE_IJVector) *vector,
                                                      (int)            *type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalSize
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setijvectorlocalsize)( long int *vector,
                                             int      *local_n,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_SetIJVectorLocalSize( (HYPRE_IJVector) *vector,
                                               (int)            *local_n  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_QueryIJVectorInsertionSem
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_queryijvectorinsertionsem)( long int *vector,
                                                  int      *level,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_QueryIJVectorInsertionSemantics(
                              (HYPRE_IJVector) *vector,
                              (int *)           level   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_InsertIJVectorRows
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_insertijvectorrows)( long int *vector,
                                           int      *n,
                                           int      *rows,
                                           double   *values,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_InsertIJVectorRows( (HYPRE_IJVector) *matrix,
                                             (int)            *n,
                                             (int *)           rows,
                                             (double *)        values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_AddRowsToIJVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_addrowstoijvector)( long int *vector,
                                          int      *n,
                                          int      *rows,
                                          double   *values,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_AddRowsToIJVector( (HYPRE_IJVector) *matrix,
                                            (int)                *n,
                                            (int *)               rows,
                                            (double *)            values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_GetIJVectorRows
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_getijvectorrows)( long int *vector,
                                        int      *row_start,
                                        int      *row_stop,
                                        double   *values,
                                        int      *ierr       )
{
   *ierr = (int) (HYPRE_GetIJVectorRows( (HYPRE_IJVector) *vector,
                                         (int)            *row_start,
                                         (int)            *row_stop,
                                         (double *)        values     ) );

}

