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
 * par_vector Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * hypre_CreateParVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_createparvector)( int      *comm,
                                        int      *global_size,
                                        int      *partitioning,
                                        long int *vector,
                                        int      *ierr          )
{
   *vector = (long int) ( hypre_CreateParVector ( (MPI_Comm) *comm,
                                                  (int)      *global_size,
                                                  (int *)     partitioning ) );
   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorDataOwner
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectordataowner)( long int *vector,
                                              int      *owns_data,
                                              int      *ierr       )
{
   *ierr = (int) ( hypre_SetParVectorDataOwner ( (hypre_ParVector *)  vector,
                                                 (int)               *owns_data ) );
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorPartitioningO
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorpartitioningo)( long int *vector,
                                                  int      *owns_partitioning,
                                                  int      *ierr    )
{
   *ierr = (int) ( hypre_SetParVectorPartitioningOwner
                         ( (HYPRE_ParCSRMatrix) *vector,
                           (int)                *owns_partitioning ) );
}

/*--------------------------------------------------------------------------
 * hypre_ReadParVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_readparvector)( long int *vector,
                                      int  *comm,
                                      char *file_name,
                                      int  *ierr       )
{
   *vector = (long int) ( hypre_ReadParVector ( (MPI_Comm) *comm,
                                                (char *)    file_name ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorConstantValue 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorconstantvalue)( long int *vector,
                                                  double   *value,
                                                  int      *ierr    )
{
   *ierr = (int) ( hypre_SetParVectorConstantValues
                      ( (hypre_ParVector *)  vector,
                        (double)            *value   ) );
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorRandomValues 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorrandomvalues)( long int *vector,
                                                 int      *seed,
                                                 int      *ierr    )
{
   *ierr = (int) ( hypre_SetParVectorRandomValues ( (hypre_ParVector *)  vector,
                                                    (int)               *seed    ) );
}

/*--------------------------------------------------------------------------
 * hypre_CopyParVector 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_copyparvector)( long int *x,
                                      long int *y,
                                      int      *ierr )
{
   *ierr = (int) ( hypre_CopyParVector ( (hypre_ParVector *) x,
                                         (hypre_ParVector *) y  ) );
}

/*--------------------------------------------------------------------------
 * hypre_ScaleParVector 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_scaleparvector)( long int *vector,
                                       double   *scale,
                                       int      *ierr )
{
   *ierr = (int) ( hypre_ScaleParVector ( (double)            *scale,
                                          (hypre_ParVector *)  vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParAxpy 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_paraxpy)( double          *a,
                                hypre_ParVector *x,
                                hypre_ParVector *y,
                                int      *ierr      )
{
   *ierr = (int) ( hypre_ParAxpy ( (double)            *a,
                                   (hypre_ParVector *)  x,
                                   (hypre_ParVector *)  y  ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParInnerProd
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parinnerprod)( long int *x,
                                     long int *y,
                                     double   *inner_prod, 
                                     int      *ierr           )
{
   *inner_prod = (double) ( hypre_ParInnerProd ( (hypre_ParVector *)  x,
                                                 (hypre_ParVector *)  y  ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_VectorToParVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_vectortoparvector)( int      *comm,
                                          long int *vector,
                                          int      *vec_starts,
                                          long int *par_vector,
                                          int      *ierr        )
{
   *par_vector = (long int) ( hypre_VectorToParVector
                                ( (MPI_Comm)       *comm,
                                  (hypre_Vector *)  vector,
                                  (int *)           vec_starts ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorToVectorAll
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectortovectorall)( long int *par_vector,
                                             long int *vector,
                                             int      *ierr        )
{
   *vector = (long int) ( hypre_ParVectorToVectorAll
                            ( (hypre_ParVector *)  par_vector ) );

   *ierr = 0;
}


