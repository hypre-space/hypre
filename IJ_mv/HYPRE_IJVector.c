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

/*--------------------------------------------------------------------------
 * HYPRE_NewIJVector
 *--------------------------------------------------------------------------*/

/*---------------------------------------------------------------------- */
/** 
"IJ" represents the "linear algebraic" conceptual view of a vector. 
The "I" and "J" in the name
are meant to be reminiscent of traditional vector notation like
A(I,J)x(J) = b(I).

Collective.

{\bf Note:} Must be the first function called using "vector" as an actual argument.
@return integer error code
@param HYPRE_IJVector &IJvector: the vector to be initialized.
@param MPI_Comm comm: a single MPI_Communicator that contains exactly the MPI processes that are to
participate in any collective operations.
@param int global_n: the dimensions of the entire, global vector.
*/
/*---------------------------------------------------------------------- */

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

int 
HYPRE_SetIJVectorPartitioning( HYPRE_IJVector  IJvector,
                               int            *partitioning )
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
      if (!hypre_IJVectorLocalStorage(vector)) hypre_NewIJVectorPar(vector);

      ierr += hypre_SetIJVectorParPartitioning(vector, partitioning);
   }
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalPartitioning
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetIJVectorLocalPartitioning( HYPRE_IJVector IJvector,
                                    int            vec_start,
                                    int            vec_stop   )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_NewIJVectorPETSc(vector);

      ierr += hypre_SetIJVectorPETScLocalPartitioning(vector,
                                                      vec_start,
                                                      vec_stop   );
   }
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_NewIJVectorISIS(vector);

      ierr += hypre_SetIJVectorISISLocalPartitioning(vector,
                                                     vec_start,
                                                     vec_stop   );
   }
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorLocalStorage(vector)) hypre_NewIJVectorPar(vector);

      ierr += hypre_SetIJVectorParLocalPartitioning(vector,
                                                    vec_start,
                                                    vec_stop   );
   }
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeIJVector
 *--------------------------------------------------------------------------*/

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
	 ierr = hypre_NewIJVectorPar(vector);
      ierr += hypre_InitializeIJVectorPar(vector);
   }
   else
      ++ierr;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributeIJVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributeIJVector( HYPRE_IJVector IJvector, int *vec_starts )
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
@param HYPRE_IJVector &vector [IN]
 the vector to be operated on. 
@param int type [IN]
possible types should be documented with each implementation. The first HYPRE implementation
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
 * HYPRE_SetIJVectorLocalSize
 *--------------------------------------------------------------------------*/

/**
Tells "vector" local size
@return integer error code
@param HYPRE_IJVector &vector [IN]
 the vector to be operated on. 
@param int local_n [IN]
 local number of components
HYPRE_SetIJVectorLocalStorageType needs to be called before this routine
*/

int 
HYPRE_SetIJVectorLocalSize( HYPRE_IJVector IJvector, int local_n )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_SetIJVectorLocalSizePETSC(vector, local_n);
   if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_SetIJVectorLocalSizeISIS(vector, local_n);
      */
/*   if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_SetIJVectorLocalSizePar(vector, local_n); 
   else */
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

/** 
Inserts a block of coefficients into an IJVector, overwriting any coefficients
in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param int *glob_vec_indices [IN]
pointer to indices of local vector to be set 
@param double value {IN]
value to which the local vector values are set

*/

int 
HYPRE_SetIJVectorLocalComponents( HYPRE_IJVector  IJvector,
                                  int             num_values,
                                  int            *glob_vec_indices,
                                  double          value)
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_SetIJVectorPETScLocalComponents(vector,
                                                   num_values,
                                                   glob_vec_indices,
                                                   value);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_SetIJVectorISISLocalComponents(vector,
                                                  num_values,
                                                  glob_vec_indices,
                                                  value);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_SetIJVectorParLocalComponents(vector,
                                                 num_values,
                                                 glob_vec_indices,
                                                 value);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

/** 
Inserts a block of coefficients into an IJVector, overwriting any coefficients
in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param int glob_vec_index_start [IN]
global index of first vector component in block to be set 
@param int glob_vec_index_stop [IN]
global index of last vector component in block to be set 
@param double value {IN]
value to which the local vector values are set

*/

int 
HYPRE_SetIJVectorLocalComponentsInBlock( HYPRE_IJVector IJvector,
                                         int            glob_vec_index_start,
                                         int            glob_vec_index_stop,
                                         double         value                 )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_SetIJVectorPETScLocalComponentsInBlock(vector,
                                                          glob_vec_index_start,
                                                          glob_vec_index_stop,
                                                          value);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_SetIJVectorISISLocalComponentsInBlock(vector,
                                                         glob_vec_index_start,
                                                         glob_vec_index_stop,
                                                         value);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_SetIJVectorParLocalComponentsInBlock(vector,
                                                        glob_vec_index_start,
                                                        glob_vec_index_stop,
                                                        value);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_InsertIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

/** 
Inserts indexed values of an array into indexed components of an IJVector, 
overwriting any components in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param int *glob_vec_indices {IN]
pointer to indices of local vector specifying which local vector components
to be overwritten
@param int *value_indices {IN]
pointer to indices of values array specifying values from which to insert 
@param double *values {IN]
pointer to array from which vector values are inserted

*/

int 
HYPRE_InsertIJVectorLocalComponents( HYPRE_IJVector  IJvector,
                                     int             num_values,
                                     int            *glob_vec_indices,
                                     int            *value_indices,
                                     double         *values            )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_InsertIJVectorPETScLocalComponents(vector,
                                                      num_values,
                                                      glob_vec_indices,
                                                      value_indices,
                                                      values);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_InsertIJVectorISISLocalComponents(vector,
                                                     num_values,
                                                     glob_vec_indices,
                                                     value_indices,
                                                     values);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_InsertIJVectorParLocalComponents(vector,
                                                    num_values,
                                                    glob_vec_indices,
                                                    value_indices,
                                                    values);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_InsertIJVectorLocalComponentsInBlock
 *--------------------------------------------------------------------------*/

/** 
Inserts indexed values of an array into a block of components of an IJVector,
overwriting any components in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param int glob_vec_index_start {IN]
@param int glob_vec_index_stop {IN]
@param int *value_indices {IN]
pointer to indices of values array specifying values from which to insert 
@param double *values {IN]
pointer to array from which vector values are inserted

*/

int 
HYPRE_InsertIJVectorLocalComponentsInBlock( HYPRE_IJVector  IJvector,
                                            int             glob_vec_index_start, 
                                            int             glob_vec_index_stop,
                                            int            *value_indices,
                                            double         *values                )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC )
      ierr = hypre_InsertIJVectorPETScLocalComponentsInBlock(vector,
                                                             glob_vec_index_start,
                                                             glob_vec_index_stop,
                                                             value_indices,
                                                             values);
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS )
      ierr = hypre_InsertIJVectorISISLocalComponentsInBlock(vector,
                                                            glob_vec_index_start,
                                                            glob_vec_index_stop,
                                                            value_indices,
                                                            values);
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR )
      ierr = hypre_InsertIJVectorParLocalComponentsInBlock(vector,
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
Modifies the values stored in vector by adding in a block.
If there is no value already in a particular vector position, the structure is augmented
with a new entry. In the event of a collision, the corresponding values are summed.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param double *values {IN]

*/

int 
HYPRE_AddToIJVectorLocalComponents( HYPRE_IJVector  IJvector,
                                    int             num_values,
                                    int            *glob_vec_indices,
                                    int            *value_indices,
                                    double         *values            )
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
Adds to indexed values of an array into a block of components of an IJVector,
overwriting any components in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param int glob_vec_index_start {IN]
@param int glob_vec_index_stop {IN]
@param int *value_indices {IN]
pointer to indices of values array specifying values from which to insert 
@param double *values {IN]
pointer to array from which vector values are inserted

*/

int 
HYPRE_AddToIJVectorLocalComponentsInBlock( HYPRE_IJVector  IJvector,
                                           int             glob_vec_index_start, 
                                           int             glob_vec_index_stop,
                                           int            *value_indices,
                                           double         *values                )
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
 * HYPRE_GetIJVectorLocalComponents
 *--------------------------------------------------------------------------*/

/** 
Puts indexed components of IJVector into indexed array of values,
overwriting any values in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param double *values {IN]

*/

int 
HYPRE_GetIJVectorLocalComponents( HYPRE_IJVector  IJvector,
                                  int             num_values,
                                  int            *glob_vec_indices,
                                  int            *value_indices,
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
Puts components in specified local block of IJVector into indexed array of values,
overwriting any values in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param int glob_vec_index_start {IN]
@param int glob_vec_index_stop {IN]
@param int *value_indices {IN]
@param double *values {IN]

*/

int 
HYPRE_GetIJVectorLocalComponentsInBlock( HYPRE_IJVector  IJvector,
                                         int             glob_vec_index_start, 
                                         int             glob_vec_index_stop,
                                         int            *value_indices,
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

@return integer error code
@param IJVector [IN]
The vector to be pointed to.
*/

void *
HYPRE_GetIJVectorLocalStorage( HYPRE_IJVector IJvector )
{
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   return( hypre_IJVectorLocalStorage(vector) );
}

/*********************************************************************************/
/* The following are routines that are not generally used by or supported for users */


/*--------------------------------------------------------------------------
 * hypre_RefIJVector
 *--------------------------------------------------------------------------*/

/**
Sets a reference to point to an IJVector.

@return integer error code
@param IJVector [IN]
The vector to be pointed to.
@param reference [OUT]
The pointer to be set to point to IJVector.
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
