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
 * hypre_ParVectorSetDataOwner
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectordataowner, HYPRE_SETPARVECTORDATAOWNER)( long int *vector,
                                              int      *owns_data,
                                              int      *ierr       )
{
   *ierr = (int) ( hypre_ParVectorSetDataOwner ( (hypre_ParVector *) *vector,
                                                 (int)               *owns_data ) );
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorPartitioningO
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorpartitioningo, HYPRE_SETPARVECTORPARTITIONINGO)( long int *vector,
                                                  int      *owns_partitioning,
                                                  int      *ierr    )
{
   *ierr = (int) ( hypre_ParVectorSetPartitioningOwner
                         ( (hypre_ParVector *) *vector,
                           (int)               *owns_partitioning ) );
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorConstantValue 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorconstantvalue, HYPRE_SETPARVECTORCONSTANTVALUE)( long int *vector,
                                                  double   *value,
                                                  int      *ierr    )
{
   *ierr = (int) ( hypre_ParVectorSetConstantValues
                      ( (hypre_ParVector *) *vector,
                        (double)            *value   ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetRandomValues 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorrandomvalues, HYPRE_SETPARVECTORRANDOMVALUES)( long int *vector,
                                                 int      *seed,
                                                 int      *ierr    )
{
   *ierr = (int) ( hypre_ParVectorSetRandomValues ( (hypre_ParVector *) *vector,
                                                    (int)               *seed    ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorCopy 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_copyparvector, HYPRE_COPYPARVECTOR)( long int *x,
                                      long int *y,
                                      int      *ierr )
{
   *ierr = (int) ( hypre_ParVectorCopy ( (hypre_ParVector *) *x,
                                         (hypre_ParVector *) *y  ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorScale 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_scaleparvector, HYPRE_SCALEPARVECTOR)( long int *vector,
                                       double   *scale,
                                       int      *ierr    )
{
   *ierr = (int) ( hypre_ParVectorScale ( (double)            *scale,
                                          (hypre_ParVector *) *vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorAxpy 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_paraxpy, HYPRE_PARAXPY)( double   *a,
                                long int *x,
                                long int *y,
                                int      *ierr )
{
   *ierr = (int) ( hypre_ParVectorAxpy ( (double)            *a,
                                   (hypre_ParVector *) *x,
                                   (hypre_ParVector *) *y  ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parinnerprod, HYPRE_PARINNERPROD)( long int *x,
                                     long int *y,
                                     double   *inner_prod, 
                                     int      *ierr           )
{
   *inner_prod = (double) ( hypre_ParVectorInnerProd ( (hypre_ParVector *) *x,
                                                 (hypre_ParVector *) *y  ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_VectorToParVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_vectortoparvector, HYPRE_VECTORTOPARVECTOR)( int      *comm,
                                          long int *vector,
                                          int      *vec_starts,
                                          long int *par_vector,
                                          int      *ierr        )
{
   *par_vector = (long int) ( hypre_VectorToParVector
                                ( (MPI_Comm)       *comm,
                                  (hypre_Vector *) *vector,
                                  (int *)           vec_starts ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorToVectorAll
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectortovectorall, HYPRE_PARVECTORTOVECTORALL)( long int *par_vector,
                                             long int *vector,
                                             int      *ierr        )
{
   *vector = (long int) ( hypre_ParVectorToVectorAll
                            ( (hypre_ParVector *) *par_vector ) );

   *ierr = 0;
}


