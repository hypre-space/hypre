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

#include "./IJ_matrix_vector.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewIJVector
 *--------------------------------------------------------------------------*/

/** 
"IJ" represents the "linear algebraic" conceptual view of a vector. 
The "I" and "J" in the name
are meant to be reminiscent of traditional vector notation like
A(I,J)x(J) = b(I).

Collective.

{\bf Note:} Must be the first function called using "vector" as an actual argument.
@return integer error code
@param HYPRE_IJVector *in_vector_ptr [IN]
the vector to be initialized.
@param MPI_Comm comm [IN]
a single MPI_Communicator that contains exactly the MPI processes that are to
participate in any collective operations.
@param int global_n [IN]
the dimensions of the entire, global vector.

*/

int HYPRE_NewIJVector( MPI_Comm comm,
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
 * HYPRE_FreeIJVector
 *--------------------------------------------------------------------------*/

/**
@return integer error code
@param HYPRE_IJVector IJvector [IN]
the vector to be freed.

*/

int 
HYPRE_FreeIJVector( HYPRE_IJVector IJvector )
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
            ierr = hypre_FreeIJVectorPETSc(vector);
         else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
            ierr = hypre_FreeIJVectorISIS(vector);
         else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
            ierr = hypre_FreeIJVectorPar(vector);
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
 * HYPRE_SetIJVectorPartitioning
 *--------------------------------------------------------------------------*/

/**
@return integer error code
@param HYPRE_IJVector IJvector
vector for which partitioning is to be set
@param int *partitioning [IN]
pointer to array of integers specifying vector decomposition across processors
HYPRE_SetIJVectorLocalStorageType needs to be called before this function.

*/

int 
HYPRE_SetIJVectorPartitioning( HYPRE_IJVector  IJvector,
                               const int      *partitioning )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_NewIJVectorPETSc(vector);

      ierr += hypre_SetIJVectorPETScPartitioning(vector,
                                                 partitioning);
   }
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_NewIJVectorISIS(vector);

      ierr += hypre_SetIJVectorISISPartitioning(vector,
                                                partitioning);
   }
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorLocalStorage(vector)) 
	 ierr = hypre_NewIJVectorPar(vector, partitioning);
      else
         ierr = hypre_SetIJVectorParPartitioning(vector, partitioning);
   }
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalPartitioning
 *--------------------------------------------------------------------------*/

/**
@return integer error code
@param HYPRE_IJVector IJvector
vector for which the local part of the partitioning is to be set
@param int vec_start_this_proc [IN]
integer specifying local index to first data element on calling processor
@param int vec_start_next_proc [IN]
integer specifying local index to first data element on next processor in
decomposition
HYPRE_SetIJVectorLocalStorageType needs to be called before this function.

*/

int 
HYPRE_SetIJVectorLocalPartitioning( HYPRE_IJVector IJvector,
                                    int            vec_start_this_proc,
                                    int            vec_start_next_proc  )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_NewIJVectorPETSc(vector);

      ierr += hypre_SetIJVectorPETScLocalPartitioning(vector,
                                                      vec_start_this_proc,
                                                      vec_start_next_proc  );
   }
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_NewIJVectorISIS(vector);

      ierr += hypre_SetIJVectorISISLocalPartitioning(vector,
                                                     vec_start_this_proc,
                                                     vec_start_next_proc  );
   }
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorLocalStorage(vector)) 
	   hypre_NewIJVectorPar(vector, NULL);

      ierr += hypre_SetIJVectorParLocalPartitioning(vector,
                                                    vec_start_this_proc,
                                                    vec_start_next_proc  );
   }
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeIJVector
 *--------------------------------------------------------------------------*/

/**
@return integer error code
@param HYPRE_IJVector IJvector
vector to be initialized
HYPRE_SetIJVectorLocalStorageType needs to be called before this function.

*/

int 
HYPRE_InitializeIJVector( HYPRE_IJVector IJvector )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr += hypre_InitializeIJVectorPETSc(vector);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr += hypre_InitializeIJVectorISIS(vector);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorLocalStorage(vector))
	 ierr = hypre_NewIJVectorPar(vector, NULL);
      ierr += hypre_InitializeIJVectorPar(vector);
   }
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributeIJVector
 *--------------------------------------------------------------------------*/

/**
@return integer error code
@param HYPRE_IJVector IJvector
vector to be distributed
@param int *vec_starts [IN]
pointer to array of integers specifying vector decomposition across processors

*/

int 
HYPRE_DistributeIJVector( HYPRE_IJVector IJvector, const int *vec_starts )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr += hypre_DistributeIJVectorPar(vector, vec_starts);
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalStorageType
 *--------------------------------------------------------------------------*/

/**
Tells "vector" which underlying "storage type" it should build from the IJVector interface
calls. Conceptually, "IJVector" is just an "interface" (an abstract base class in C++-speak),
with no implementation. Specific concrete classes implement this interface. One way for a user
to choose which concrete class to use is by *instantiating* the class of choice (in an OO
implementation) or by *linking* in the implementation of choice (a "compile time" choice). An
intermediate method is through the use of HYPRE_SetIJVectorStorageType that *may* be provided
by specific implementations. For example, it is possible to build either a PETSc MPIAIJ vector
or a ParVector from the IJ vector interface, and this choice can be controlled through this
function.

Not collective, but must be same on all processors in group that stores vector.
@return integer error code
@param HYPRE_IJVector IJvector
vector for which the type is to be set
@param int type [IN]
Possible types should be documented with each implementation. The first HYPRE implementation
will list its possible values in HYPRE.h (at least for now). Note that this function is optional,
as all implementations must have a default. Eventually the plan is to let solvers choose a storage
type that they can work with automatically, but this is in the future.

*/

int 
HYPRE_SetIJVectorLocalStorageType( HYPRE_IJVector IJvector, int type )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   hypre_IJVectorLocalStorageType(vector) = type;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_ZeroIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

/** 
zeros all of the local vector components, overwriting
all indexed coefficients.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector
vector, some components of which are to be set
@param int *glob_vec_indices [IN]
pointer to global indices of local vector components to be set 
@param double value [IN]
value to which the local vector components are to be set

*/

int 
HYPRE_ZeroIJVectorLocalComponents( HYPRE_IJVector  IJvector)
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_ZeroIJVectorPETScLocalComponents(vector);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_ZeroIJVectorISISLocalComponents(vector);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_ZeroIJVectorParLocalComponents(vector);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

/** 
From indexed values of an array, sets indexed components of an IJVector, 
overwriting all indexed components.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
vector, some components of which are to be inserted to
@param int *glob_vec_indices [IN]
pointer to indices of local vector specifying which local vector components
to be overwritten
@param int *value_indices [IN]
pointer to indices of values array specifying values from which to insert 
@param double *values [IN]
pointer to array from which vector values are inserted

*/

int 
HYPRE_SetIJVectorLocalComponents( HYPRE_IJVector  IJvector,
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
      ierr = hypre_SetIJVectorParLocalComponents(vector,
                                                 num_values,
                                                 glob_vec_indices,
                                                 value_indices,
                                                 values);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

/** 
From indexed values of an array, sets a block of components of an IJVector,
overwriting all components of the block.

Not collective.

@return integer error code
@param HYPRE_IJVector IJvector
the vector to be operated on
@param int glob_vec_index_start [IN]
global index of first vector component in block to be inserted 
@param int glob_vec_index_stop [IN]
global index of last vector component in block to be set 
@param int *value_indices [IN]
pointer to indices of values array specifying values from which to insert 
@param double *values [IN]
pointer to array from which vector values are inserted

*/

int 
HYPRE_SetIJVectorLocalComponentsInBlock( HYPRE_IJVector  IJvector,
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
      ierr = hypre_SetIJVectorParLocalComponentsInBlock(vector,
                                                        glob_vec_index_start,
                                                        glob_vec_index_stop,
                                                        value_indices,
                                                        values);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_AddToIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

/** 
Adds indexed values of an array to an indexed set of components of
an IJVector, overwriting all indexed components.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
vector, some components of which are to be summed to
@param int *glob_vec_indices [IN]
pointer to global indices of local vector components to be set 
@param int *value_indices [IN]
pointer to indices of value array members to be summed to local vector
components
@param double *values [IN]
pointer to array from which values are taken for summing

*/

int 
HYPRE_AddToIJVectorLocalComponents( HYPRE_IJVector  IJvector,
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
      ierr = hypre_AddToIJVectorParLocalComponents(vector,
                                                   num_values,
                                                   glob_vec_indices,
                                                   value_indices,
                                                   values);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_AddToIJVectorLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

/** 
Adds indexed values of an array into a block of components of an IJVector,
overwriting all components in the block.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector
vector, the components of which are to be summed to
@param int glob_vec_index_start [IN]
global index of first vector component in block to be summed to
@param int glob_vec_index_stop [IN]
global index of last vector component in block to be summed to
@param int *value_indices [IN]
pointer to indices of value array members to be summed to local vector
components
@param double *values [IN]
pointer to array from which values are taken for summing

*/

int 
HYPRE_AddToIJVectorLocalComponentsInBlock( HYPRE_IJVector  IJvector,
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
      ierr = hypre_AddToIJVectorParLocalComponentsInBlock(vector,
                                                          glob_vec_index_start,
                                                          glob_vec_index_stop,
                                                          value_indices,
                                                          values);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleIJVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleIJVector( HYPRE_IJVector  IJvector )
{
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      return( hypre_AssembleIJVectorPETSc( vector ) );
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      return( hypre_AssembleIJVectorISIS( vector ) );
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      return(  hypre_AssembleIJVectorPar( vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

/** 
Puts indexed components of IJVector into indexed array of values,
overwriting any values in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
vector from which components are to be fetched
@param double *values

*/

int 
HYPRE_GetIJVectorLocalComponents( HYPRE_IJVector  IJvector,
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
      ierr = hypre_GetIJVectorParLocalComponents(vector,
                                                 num_values,
                                                 glob_vec_indices,
                                                 value_indices,
                                                 values);
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_GetIJVectorLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

/** 
Puts components in specified local block of IJVector into indexed array of values, overwriting any values in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector from which components are to be fetched 
@param int glob_vec_index_start [IN]
@param int glob_vec_index_stop [IN]
@param int *value_indices [IN]
@param double *values

*/

int 
HYPRE_GetIJVectorLocalComponentsInBlock( HYPRE_IJVector  IJvector,
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
      ierr = hypre_GetIJVectorParLocalComponentsInBlock(vector,
                                                        glob_vec_index_start,
                                                        glob_vec_index_stop,
                                                        value_indices,
                                                        values);
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_GetIJVectorLocalStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_GetIJVectorLocalStorageType( HYPRE_IJVector IJvector, int *type )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   if (vector)
      *type = hypre_IJVectorLocalStorageType(vector);
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_GetIJVectorLocalStorage
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to an underlying vector type used to implement IJVector.
Assumes that the implementation has an underlying vector, so it would not
work with a direct implementation of IJVector. 

@return void pointer to local storage
@param IJVector [IN]

*/

void *
HYPRE_GetIJVectorLocalStorage( HYPRE_IJVector IJvector )
{
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   return( hypre_IJVectorLocalStorage(vector) );
}

/*********************************************************************************/
/* The following are functions not generally used by or supported for users */


/*--------------------------------------------------------------------------
 * hypre_RefIJVector
 *--------------------------------------------------------------------------*/

/**
Sets a reference to point to an IJVector.

@return integer error code
@param IJVector [IN]
vector to be pointed at
@param reference [OUT]
pointer to IJVector

*/

int 
hypre_RefIJVector( HYPRE_IJVector IJvector, HYPRE_IJVector *reference )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   hypre_IJVectorReferenceCount(vector) ++;

   *reference = IJvector;

   return(ierr);
}
