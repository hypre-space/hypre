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

#include "./IJ_mv.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixcreate, HYPRE_IJMATRIXCREATE)( int      *comm,
                                    long int *matrix,
                                    int      *global_m,
                                    int      *global_n,
                                    int      *ierr      )
{
   *ierr = (int) ( HYPRE_IJMatrixCreate( (MPI_Comm)      *comm,
                                      (HYPRE_IJMatrix *)  matrix,
                                      (int)              *global_m,
                                      (int)              *global_n  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixdestroy, HYPRE_IJMATRIXDESTROY)( long int *matrix,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixDestroy( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixinitialize, HYPRE_IJMATRIXINITIALIZE)( long int *matrix,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixInitialize( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixassemble, HYPRE_IJMATRIXASSEMBLE)( long int *matrix,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixAssemble ( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDistribute
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixdistribute, HYPRE_IJMATRIXDISTRIBUTE)( long int *matrix,
                                           int      *row_starts,
                                           int      *col_starts,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_IJMatrixDistribute( (HYPRE_IJMatrix) *matrix,
                                             (int *)           row_starts,
                                             (int *)           col_starts  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJMatrixLocalStorageTy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixsetlocalstoragety, HYPRE_IJMATRIXSETLOCALSTORAGETY)( long int *matrix,
                                                  int      *type,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixSetLocalStorageType( (HYPRE_IJMatrix) *matrix,
                                                      (int)            *type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetLocalSize
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixsetlocalsize, HYPRE_IJMATRIXSETLOCALSIZE)( long int *matrix,
                                             int      *local_m,
                                             int      *local_n,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_IJMatrixSetLocalSize( (HYPRE_IJMatrix) *matrix,
                                               (int)            *local_m,
                                               (int)            *local_n  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetRowSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetrowsizes, HYPRE_IJMATRIXSETROWSIZES)( long int *matrix,
                                            int      *sizes,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixSetRowSizes( (HYPRE_IJMatrix) *matrix,
                                              (const int *)     sizes   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetDiagRowSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetdiagrowsizes, HYPRE_IJMATRIXSETDIAGROWSIZES)( long int *matrix,
                                                int      *sizes,
                                                int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixSetDiagRowSizes( (HYPRE_IJMatrix) *matrix,
                                                  (const int *)     sizes   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetOffDiagRowSize
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetoffdiagrowsize, HYPRE_IJMATRIXSETOFFDIAGROWSIZE)( long int *matrix,
                                                  int      *sizes,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixSetOffDiagRowSizes( (HYPRE_IJMatrix) *matrix,
                                                     (const int *)     sizes   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixQueryInsertionSem
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixqueryinsertionsem, HYPRE_IJMATRIXQUERYINSERTIONSEM)( long int *matrix,
                                                  int      *level,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixQueryInsertionSemantics(
                              (HYPRE_IJMatrix) *matrix,
                              (int *)           level   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInsertBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixinsertblock, HYPRE_IJMATRIXINSERTBLOCK)( long int *matrix,
                                            int      *m,
                                            int      *n,
                                            int      *rows,
                                            int      *cols,
                                            double   *values,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixInsertBlock( (HYPRE_IJMatrix) *matrix,
                                              (int)            *m,
                                              (int)            *n,
                                              (const int *)     rows,
                                              (const int *)     cols,
                                              (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixaddtoblock, HYPRE_IJMATRIXADDTOBLOCK)( long int *matrix,
                                           int      *m,
                                           int      *n,
                                           int      *rows,
                                           int      *cols,
                                           double   *values,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixAddToBlock( (HYPRE_IJMatrix) *matrix,
                                             (int)            *m,
                                             (int)            *n,
                                             (const int *)     rows,
                                             (const int *)     cols,
                                             (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInsertRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixinsertrow, HYPRE_IJMATRIXINSERTROW)( long int *matrix,
                                          int      *n,
                                          int      *row,
                                          int      *cols,
                                          double   *values,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixInsertRow( (HYPRE_IJMatrix) *matrix,
                                            (int)            *n,
                                            (int)            *row,
                                            (const int *)     cols,
                                            (const double *)  values   ) );

} 

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixaddtorow, HYPRE_IJMATRIXADDTOROW)( long int *matrix,
                                           int      *n,
                                           int      *row,
                                           int      *cols,
                                           double   *values,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixAddToRow( (HYPRE_IJMatrix) *matrix,
                                             (int)            *n,
                                             (int)            *row,
                                             (const int *)     cols,
                                             (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToRowAfter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixaddtorowafter, HYPRE_IJMATRIXADDTOROWAFTER)( long int *matrix,
                                           int      *n,
                                           int      *row,
                                           int      *cols,
                                           double   *values,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixAddToRowAfter( (HYPRE_IJMatrix) *matrix,
                                             (int)            *n,
                                             (int)            *row,
                                             (const int *)     cols,
                                             (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixsetvalues, HYPRE_IJMATRIXSETVALUES)( long int *matrix,
                                           int      *n,
                                           int      *row,
                                           int      *cols,
                                           double   *values,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixSetValues( (HYPRE_IJMatrix) *matrix,
                                             (int)            *n,
                                             (int)            *row,
                                             (const int *)     cols,
                                             (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixaddtovalues, HYPRE_IJMATRIXADDTOVALUES)( long int *matrix,
                                           int      *n,
                                           int      *row,
                                           int      *cols,
                                           double   *values,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixAddToValues( (HYPRE_IJMatrix) *matrix,
                                             (int)            *n,
                                             (int)            *row,
                                             (const int *)     cols,
                                             (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetBlockValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixsetblockvalues, HYPRE_IJMATRIXSETBLOCKVALUES)( long int *matrix,
                                           int      *m,
                                           int      *n,
                                           int      *rows,
                                           int      *cols,
                                           double   *values,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixSetBlockValues( (HYPRE_IJMatrix) *matrix,
                                             (int)            *m,
                                             (int)            *n,
                                             (const int *)     rows,
                                             (const int *)     cols,
                                             (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToBlockValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixaddtoblockvalues, HYPRE_IJMATRIXADDTOBLOCKVALUES)( long int *matrix,
                                           int      *m,
                                           int      *n,
                                           int      *rows,
                                           int      *cols,
                                           double   *values,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixAddToBlockValues( (HYPRE_IJMatrix) *matrix,
                                             (int)            *m,
                                             (int)            *n,
                                             (const int *)     rows,
                                             (const int *)     cols,
                                             (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetLocalStorage
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetlocalstorage, HYPRE_IJMATRIXGETLOCALSTORAGE)( long int *matrix,
                                                long int *local_storage,
                                                int      *ierr    )
{
   *ierr = 0;

   *local_storage = (long int) ( HYPRE_IJMatrixGetLocalStorage(
                                       (HYPRE_IJMatrix) *matrix ) );

   if (!(*local_storage)) ++(*ierr); 
}
