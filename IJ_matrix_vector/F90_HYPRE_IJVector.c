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
 * HYPRE_IJVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorcreate)( int      *comm,
                                    long int *vector,
                                    int      *global_n,
                                    int      *ierr      )
{
   *ierr = (int) ( HYPRE_IJVectorCreate( (MPI_Comm)         *comm,
                                      (HYPRE_IJVector *)  vector,
                                      (int)              *global_n  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectordestroy)( long int *vector,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorDestroy( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetPartitioning
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectorsetpartitioning)( long int *vector,
                                                int      *partitioning,
                                                int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorSetPartitioning( (HYPRE_IJVector) *vector,
                                                  (int *)           partitioning ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetLocalPartitioning
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectorsetlocalpartition)( long int *vector,
                                                  int      *vec_start_this_proc,
                                                  int      *vec_start_next_proc,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorSetLocalPartitioning( (HYPRE_IJVector) *vector,
                                                       (int)            *vec_start_this_proc,
                                                       (int)            *vec_start_next_proc ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorinitialize)( long int *vector,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorInitialize( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorDistribute
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectordistribute)( long int *vector,
                                           int      *vec_starts,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_IJVectorDistribute( (HYPRE_IJVector) *vector,
                                             (int *)           vec_starts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetLocalStorageType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectorsetlocalstoragety)( long int *vector,
                                                  int      *type,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorSetLocalStorageType( (HYPRE_IJVector) *vector,
                                                      (int)            *type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorZeroLocalComponents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorzerolocalcomps)( long int *vector,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorZeroLocalComponents(
                         (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetLocalComponents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorsetlocalcomps)( long int *vector,
                                           int      *num_values,
                                           int      *glob_vec_indices,
                                           int      *value_indices,
                                           double   *values,
                                           int      *ierr              )
{
   *ierr = (int) ( HYPRE_IJVectorSetLocalComponents(
                         (HYPRE_IJVector) *vector,
                         (int)            *num_values,
                         (int *)           glob_vec_indices,
                         (int *)           value_indices,
                         (double *)        values            ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorsetlocalcompsinbl)( long int *vector,
                                                  int      *glob_vec_start,
                                                  int      *glob_vec_stop,
                                                  int      *value_indices,
                                                  double   *values,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_IJVectorSetLocalComponentsInBlock(
                          (HYPRE_IJVector) *vector,
                          (int)            *glob_vec_start,
                          (int)            *glob_vec_stop,
                          (int *)           value_indices,
                          (double *)        values          ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAddToLocalComps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectoraddtolocalcomps)( long int *vector,
                                             int      *num_values,
                                             int      *glob_vec_indices,
                                             int      *value_indices,
                                             double   *values,
                                             int      *ierr              )
{
   *ierr = (int) ( HYPRE_IJVectorAddToLocalComponents(
                         (HYPRE_IJVector) *vector,
                         (int)            *num_values,
                         (int *)           glob_vec_indices,
                         (int *)           value_indices,
                         (double *)        values          ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAddToLocalCompsBl
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectoraddtolocalcompsbl)( long int *vector,
                                                  int      *glob_vec_start,
                                                  int      *glob_vec_stop,
                                                  int      *value_indices,
                                                  double   *values,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_IJVectorAddToLocalComponentsInBlock(
                         (HYPRE_IJVector) *vector,
                         (int)            *glob_vec_start,
                         (int)            *glob_vec_stop,
                         (int *)           value_indices,
                         (double *)        values          ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorassemble)( long int *vector,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorAssemble( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalComponents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetlocalcomps)( long int *vector,
                                           int      *num_values,
                                           int      *glob_vec_indices,
                                           int      *value_indices,
                                           double   *values,
                                           int      *ierr              )
{
   *ierr = (int) ( HYPRE_IJVectorGetLocalComponents(
                         (HYPRE_IJVector) *vector,
                         (int)            *num_values,
                         (int *)           glob_vec_indices,
                         (int *)           value_indices,
                         (double *)        values          ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetlocalcompsinbl)( long int *vector,
                                                  int      *glob_vec_start,
                                                  int      *glob_vec_stop,
                                                  int      *value_indices,
                                                  double   *values,
                                                  int      *ierr            )
{
   *ierr = (int) (HYPRE_IJVectorGetLocalComponentsInBlock(
                        (HYPRE_IJVector) *vector,
                        (int)            *glob_vec_start,
                        (int)            *glob_vec_stop,
                        (int *)           value_indices,
                        (double *)        values          ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalStorageType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetlocalstoragety)( long int *vector,
                                                  int      *type,
                                                  int      *ierr    )
{
   *ierr = (int) (HYPRE_IJVectorGetLocalStorageType(
                        (HYPRE_IJVector) *vector,
                        (int *)           type    ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalStorage
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetlocalstorage)( long int *vector,
                                                long int *local_storage,
                                                int      *ierr           )
{
   *ierr = 0;

   *local_storage = (long int) (HYPRE_IJVectorGetLocalStorage(
                                      (HYPRE_IJVector) *vector ) );

   if (!(*local_storage)) ++(*ierr);
}


