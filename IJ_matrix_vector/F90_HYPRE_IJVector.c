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
 * HYPRE_SetIJVectorPartitioning
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setijvectorpartitioning)( long int *vector,
                                                int      *partitioning,
                                                int      *ierr    )
{
   *ierr = (int) ( HYPRE_SetIJVectorPartitioning( (HYPRE_IJVector) *vector,
                                                  (int *)           partitioning ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalPartitioning
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setijvectorlocalpartition)( long int *vector,
                                                  int      *vec_start_this_proc,
                                                  int      *vec_start_next_proc,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_SetIJVectorLocalPartitioning( (HYPRE_IJVector) *vector,
                                                       (int)             vec_start_this_proc,
                                                       (int)             vec_start_next_proc ) );
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
 * HYPRE_DistributeIJVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_distributeijvector)( long int *vector,
                                           int      *vec_starts,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_DistributeIJVector( (HYPRE_IJVector) *vector,
                                             (int *)           vec_starts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalStorageType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setijvectorlocalstoragety)( long int *vector,
                                                  int      *type,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_SetIJVectorLocalStorageType( (HYPRE_IJVector) *vector,
                                                      (int)            *type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ZeroIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_zeroijveclocalcomps)( long int *vector,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ZeroIJVectorLocalComponents(
                         (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_setijveclocalcomps)( long int *vector,
                                           int      *num_values,
                                           int      *glob_vec_indices,
                                           int      *value_indices,
                                           double   *values,
                                           int      *ierr              )
{
   *ierr = (int) ( HYPRE_SetIJVectorLocalComponents(
                         (HYPRE_IJVector) *vector,
                         (int)            *num_values,
                         (int *)           glob_vec_indices,
                         (int *)           value_indices,
                         (double *)        values            ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_setijveclocalcompsinblock)( long int *vector,
                                                  int      *glob_vec_start,
                                                  int      *glob_vec_stop,
                                                  int      *value_indices,
                                                  double   *values,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_SetIJVectorLocalComponentsInBlock(
                          (HYPRE_IJVector) *vector,
                          (int)            *glob_vec_start,
                          (int)            *glob_vec_stop,
                          (int *)           value_indices,
                          (double *)        values          ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AddToIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_addtoijveclocalcomps)( long int *vector,
                                             int      *num_values,
                                             int      *glob_vec_indices,
                                             int      *value_indices,
                                             double   *values,
                                             int      *ierr              )
{
   *ierr = (int) ( HYPRE_AddToIJVectorLocalComponents(
                         (HYPRE_IJVector) *vector,
                         (int)            *num_values,
                         (int *)           glob_vec_indices,
                         (int *)           value_indices,
                         (double *)        values          ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AddToIJVectorLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_addtoijveclocalcompsinblo)( long int *vector,
                                                  int      *glob_vec_start,
                                                  int      *glob_vec_stop,
                                                  int      *value_indices,
                                                  double   *values,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_AddToIJVectorLocalComponentsInBlock(
                         (HYPRE_IJVector) *vector,
                         (int)            *glob_vec_start,
                         (int)            *glob_vec_stop,
                         (int *)           value_indices,
                         (double *)        values          ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleIJVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_assembleijvector)( long int *vector,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_AssembleIJVector( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_getijveclocalcomps)( long int *vector,
                                           int      *num_values,
                                           int      *glob_vec_indices,
                                           int      *value_indices,
                                           double   *values,
                                           int      *ierr              )
{
   *ierr = (int) ( HYPRE_GetIJVectorLocalComponents(
                         (HYPRE_IJVector) *vector,
                         (int)            *num_values,
                         (int *)           glob_vec_indices,
                         (int *)           value_indices,
                         (double *)        values          ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetIJVectorLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_getijveclocalcompsinblock)( long int *vector,
                                                  int      *glob_vec_start,
                                                  int      *glob_vec_stop,
                                                  int      *value_indices,
                                                  double   *values,
                                                  int      *ierr            )
{
   *ierr = (int) (HYPRE_GetIJVectorLocalComponentsInBlock(
                        (HYPRE_IJVector) *vector,
                        (int)            *glob_vec_start,
                        (int)            *glob_vec_stop,
                        (int *)           value_indices,
                        (double *)        values          ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_GetIJVectorLocalStorageType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_getijvectorlocalstoragety)( long int *vector,
                                                  int      *type,
                                                  int      *ierr    )
{
   *ierr = (int) (HYPRE_GetIJVectorLocalStorageType(
                        (HYPRE_IJVector) *vector,
                        (int *)           type    ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_GetIJVectorLocalStorage
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_getijvectorlocalstorage)( long int *vector,
                                                long int *local_storage,
                                                int      *ierr           )
{
   *ierr = 0;

   *local_storage = (long int) (HYPRE_GetIJVectorLocalStorage(
                                      (HYPRE_IJVector) *vector ) );

   if (!(*local_storage)) ++(*ierr);
}


