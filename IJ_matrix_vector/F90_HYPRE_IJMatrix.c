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
 * HYPRE_NewIJMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_newijmatrix)( int      *comm,
                                    long int *matrix,
                                    int      *global_m,
                                    int      *global_n,
                                    int      *ierr      )
{
   *ierr = (int) ( HYPRE_NewIJMatrix( (MPI_Comm)         *comm,
                                      (HYPRE_IJMatrix *)  matrix,
                                      (int)              *global_m,
                                      (int)              *global_n  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeIJMatrix
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_freeijmatrix)( long int *matrix,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_FreeIJMatrix( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeIJMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_initializeijmatrix)( long int *matrix,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_InitializeIJMatrix( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleIJMatrix
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_assembleijmatrix)( long int *matrix,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_AssembleIJMatrix ( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributeIJMatrix
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_distributeijmatrix)( long int *matrix,
                                           int      *row_starts,
                                           int      *col_starts,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_DistributeIJMatrix( (HYPRE_IJMatrix) *matrix,
                                             (int *)           row_starts,
                                             (int *)           col_starts  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJMatrixLocalStorageT
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setijmatrixlocalstoragety)( long int *matrix,
                                                  int      *type,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_SetIJMatrixLocalStorageType( (HYPRE_IJMatrix) *matrix,
                                                      (int)            *type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJMatrixLocalSize
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setijmatrixlocalsize)( long int *matrix,
                                             int      *local_m,
                                             int      *local_n,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_SetIJMatrixLocalSize( (HYPRE_IJMatrix) *matrix,
                                               (int)            *local_m,
                                               (int)            *local_n  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJMatrixRowSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_setijmatrixrowsizes)( long int *matrix,
                                            int      *sizes,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_SetIJMatrixRowSizes( (HYPRE_IJMatrix) *matrix,
                                              (int *)           sizes   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJMatrixDiagRowSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_setijmatrixdiagrowsizes)( long int *matrix,
                                                int      *sizes,
                                                int      *ierr    )
{
   *ierr = (int) ( HYPRE_SetIJMatrixDiagRowSizes( (HYPRE_IJMatrix) *matrix,
                                                  (int *)           sizes   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJMatrixOffDiagRowSize
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_setijmatrixoffdiagrowsize)( long int *matrix,
                                                  int      *sizes,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_SetIJMatrixOffDiagRowSizes( (HYPRE_IJMatrix) *matrix,
                                                     (int *)           sizes   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_QueryIJMatrixInsertionSem
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_queryijmatrixinsertionsem)( long int *matrix,
                                                  int      *level,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_QueryIJMatrixInsertionSemantics(
                              (HYPRE_IJMatrix) *matrix,
                              (int *)           level   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_InsertIJMatrixBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_insertijmatrixblock)( long int *matrix,
                                            int      *m,
                                            int      *n,
                                            int      *rows,
                                            int      *cols,
                                            double   *values,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_InsertIJMatrixBlock( (HYPRE_IJMatrix) *matrix,
                                              (int)            *m,
                                              (int)            *n,
                                              (int *)           rows,
                                              (int *)           cols,
                                              (double *)        values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_AddBlockToIJMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_addblocktoijmatrix)( long int *matrix,
                                           int      *m,
                                           int      *n,
                                           int      *rows,
                                           int      *cols,
                                           double   *values,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_AddBlockToIJMatrix( (HYPRE_IJMatrix) *matrix,
                                             (int)            *m,
                                             (int)            *n,
                                             (int *)           rows,
                                             (int *)           cols,
                                             (double *)        values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_InsertIJMatrixRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_insertijmatrixrow)( long int *matrix,
                                          int      *n,
                                          int      *row,
                                          int      *cols,
                                          double   *values,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_InsertIJMatrixRow( (HYPRE_IJMatrix) *matrix,
                                            (int)            *n,
                                            (int)            *row,
                                            (int *)           cols,
                                            (double *)        values   ) );

} 

/*--------------------------------------------------------------------------
 * HYPRE_GetIJMatrixLocalStorage
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_getijmatrixlocalstorage)( long int *matrix,
                                                long int *local_storage,
                                                int      *ierr    )
{
   *ierr = 0;

   *local_storage = (long int) ( HYPRE_GetIJMatrixLocalStorage(
                                       (HYPRE_IJMatrix) *matrix ) );

   if (!(*local_storage)) ++(*ierr); 
}
