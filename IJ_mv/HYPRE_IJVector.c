/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_IJVector interface
 *
 *****************************************************************************/

#include "./IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorCreate
 *--------------------------------------------------------------------------*/

int HYPRE_IJVectorCreate( MPI_Comm comm,
                       HYPRE_IJVector *in_vector_ptr, 
                       int global_n)

{
   int ierr=0;

   hypre_IJVector    *vector;

   vector = hypre_CTAlloc(hypre_IJVector, 1);

   hypre_IJVectorContext(vector) = comm;
   hypre_IJVectorN(vector)       = global_n;
   hypre_IJVectorLocalStorage(vector) = NULL;
   hypre_IJVectorLocalStorageType(vector) = HYPRE_UNITIALIZED;
   hypre_IJVectorReferenceCount(vector) = 1;

   *in_vector_ptr = (HYPRE_IJVector) vector;
  
   return( ierr ); 
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorDestroy( HYPRE_IJVector IJvector )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   if (vector)
   {
      hypre_IJVectorReferenceCount(vector) --;
   
      if ( hypre_IJVectorReferenceCount(vector) <= 0 )
      {
	/*
         if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
            ierr = hypre_IJVectorDestroyPETSc(vector);
         else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
            ierr = hypre_IJVectorDestroyISIS(vector);
         else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
            ierr = hypre_IJVectorDestroyPar(vector);
         else
            ierr = -1;

         hypre_TFree(vector);
      }
   }
   else
   {
      ierr = -1;
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetPartitioning
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorSetPartitioning( HYPRE_IJVector  IJvector,
                               const int      *partitioning )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_CreateIJVectorPETSc(vector);

      ierr += hypre_SetIJVectorPETScPartitioning(vector,
                                                 partitioning);
   }
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_CreateIJVectorISIS(vector);

      ierr += hypre_SetIJVectorISISPartitioning(vector,
                                                partitioning);
   }
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorLocalStorage(vector)) 
	 ierr = hypre_IJVectorCreatePar(vector, partitioning);
      else
         ierr = hypre_IJVectorSetPartitioningPar(vector, partitioning);
   }
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetLocalPartitioning
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorSetLocalPartitioning( HYPRE_IJVector IJvector,
                                    int            vec_start_this_proc,
                                    int            vec_start_next_proc  )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_CreateIJVectorPETSc(vector);

      ierr += hypre_SetIJVectorPETScLocalPartitioning(vector,
                                                      vec_start_this_proc,
                                                      vec_start_next_proc  );
   }
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_CreateIJVectorISIS(vector);

      ierr += hypre_SetIJVectorISISLocalPartitioning(vector,
                                                     vec_start_this_proc,
                                                     vec_start_next_proc  );
   }
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorLocalStorage(vector)) 
	   hypre_IJVectorCreatePar(vector, NULL);

      ierr += hypre_IJVectorSetLocalPartitioningPar(vector,
                                                    vec_start_this_proc,
                                                    vec_start_next_proc  );
   }
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorInitialize( HYPRE_IJVector IJvector )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr += hypre_IJVectorInitializePETSc(vector);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr += hypre_IJVectorInitializeISIS(vector);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorLocalStorage(vector))
	 ierr = hypre_IJVectorCreatePar(vector, NULL);
      ierr += hypre_IJVectorInitializePar(vector);
   }
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorDistribute
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJVectorDistribute( HYPRE_IJVector IJvector, const int *vec_starts )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr += hypre_IJVectorDistributePar(vector, vec_starts);
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetLocalStorageType
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorSetLocalStorageType( HYPRE_IJVector IJvector, int type )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   hypre_IJVectorLocalStorageType(vector) = type;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorZeroLocalComponents
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorZeroLocalComponents( HYPRE_IJVector  IJvector)
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_ZeroIJVectorPETScLocalComponents(vector);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_ZeroIJVectorISISLocalComponents(vector);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_IJVectorZeroLocalComponentsPar(vector);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetLocalComponents
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorSetLocalComponents( HYPRE_IJVector  IJvector,
                                  int             num_values,
                                  const int      *glob_vec_indices,
                                  const int      *value_indices,
                                  const double   *values            )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_SetIJVectorPETScLocalComponents(vector,
                                                   num_values,
                                                   glob_vec_indices,
                                                   value_indices,
                                                   values);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_SetIJVectorISISLocalComponents(vector,
                                                  num_values,
                                                  glob_vec_indices,
                                                  value_indices,
                                                  values);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_IJVectorSetLocalComponentsPar(vector,
                                                 num_values,
                                                 glob_vec_indices,
                                                 value_indices,
                                                 values);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorSetLocalComponentsInBlock( HYPRE_IJVector  IJvector,
                                         int             glob_vec_index_start, 
                                         int             glob_vec_index_stop,
                                         const int      *value_indices,
                                         const double   *values                )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_SetIJVectorPETScLocalComponentsInBlock(vector,
                                                          glob_vec_index_start,
                                                          glob_vec_index_stop,
                                                          value_indices,
                                                          values);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_SetIJVectorISISLocalComponentsInBlock(vector,
                                                         glob_vec_index_start,
                                                         glob_vec_index_stop,
                                                         value_indices,
                                                         values);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_IJVectorSetLocalComponentsInBlockPar(vector,
                                                        glob_vec_index_start,
                                                        glob_vec_index_stop,
                                                        value_indices,
                                                        values);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAddToLocalComponents
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorAddToLocalComponents( HYPRE_IJVector  IJvector,
                                    int             num_values,
                                    const int      *glob_vec_indices,
                                    const int      *value_indices,
                                    const double   *values            )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_AddToIJVectorPETScLocalComponents(vector,
                                                     num_values,
                                                     glob_vec_indices,
                                                     value_indices,
                                                     values);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_AddToIJVectorISISLocalComponents(vector,
                                                    num_values,
                                                    glob_vec_indices,
                                                    value_indices,
                                                    values);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_IJVectorAddToLocalComponentsPar(vector,
                                                   num_values,
                                                   glob_vec_indices,
                                                   value_indices,
                                                   values);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAddToLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorAddToLocalComponentsInBlock( HYPRE_IJVector  IJvector,
                                           int             glob_vec_index_start, 
                                           int             glob_vec_index_stop,
                                           const int      *value_indices,
                                           const double   *values                )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_AddToIJVectorPETScLocalComponentsInBlock(vector,
                                                            glob_vec_index_start,
                                                            glob_vec_index_stop,
                                                            value_indices,
                                                            values);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_AddToIJVectorISISLocalComponentsInBlock(vector,
                                                           glob_vec_index_start,
                                                           glob_vec_index_stop,
                                                           value_indices,
                                                           values);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_IJVectorAddToLocalComponentsInBlockPar(vector,
                                                          glob_vec_index_start,
                                                          glob_vec_index_stop,
                                                          value_indices,
                                                          values);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorAssemble( HYPRE_IJVector  IJvector )
{
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      return( hypre_AssembleIJVectorPETSc( vector ) );
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      return( hypre_AssembleIJVectorISIS( vector ) );
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      return(  hypre_IJVectorAssemblePar( vector ) );
   else 
      return(0);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalComponents
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorGetLocalComponents( HYPRE_IJVector  IJvector,
                                  int             num_values,
                                  const int      *glob_vec_indices,
                                  const int      *value_indices,
                                  double         *values            )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_GetIJVectorPETScLocalComponents(vector,
                                                   num_values,
                                                   glob_vec_indices,
                                                   value_indices,
                                                   values);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_GetIJVectorISISLocalComponents(vector,
                                                  num_values,
                                                  glob_vec_indices,
                                                  value_indices,
                                                  values);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_IJVectorGetLocalComponentsPar(vector,
                                                 num_values,
                                                 glob_vec_indices,
                                                 value_indices,
                                                 values);
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorGetLocalComponentsInBlock( HYPRE_IJVector  IJvector,
                                         int             glob_vec_index_start, 
                                         int             glob_vec_index_stop,
                                         const int      *value_indices,
                                         double         *values                )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_GetIJVectorPETScLocalComponentsInBlock(vector,
                                                          glob_vec_index_start,
                                                          glob_vec_index_stop,
                                                          value_indices,
                                                          values);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_GetIJVectorISISLocalComponentsInBlock(vector,
                                                         glob_vec_index_start,
                                                         glob_vec_index_stop,
                                                         value_indices,
                                                         values);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_IJVectorGetLocalComponentsInBlockPar(vector,
                                                        glob_vec_index_start,
                                                        glob_vec_index_stop,
                                                        value_indices,
                                                        values);
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_IJVectorGetLocalStorageType( HYPRE_IJVector IJvector, int *type )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   if (vector)
      *type = hypre_IJVectorLocalStorageType(vector);
   else
      ++ierr;

   return(ierr);
}

/*********************************************************************************/
/* The following are functions not generally used by or supported for users */


/*--------------------------------------------------------------------------
 * hypre_RefIJVector
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalStorage
 *--------------------------------------------------------------------------*/

void *
HYPRE_IJVectorGetLocalStorage( HYPRE_IJVector IJvector )
{
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   return( hypre_IJVectorLocalStorage(vector) );
}


int 
hypre_RefIJVector( HYPRE_IJVector IJvector, HYPRE_IJVector *reference )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   hypre_IJVectorReferenceCount(vector) ++;

   *reference = IJvector;

   return(ierr);
}
