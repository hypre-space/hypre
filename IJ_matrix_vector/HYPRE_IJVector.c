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
@param int local_n: the dimensions of the locally stored vector.
*/
/*---------------------------------------------------------------------- */

int HYPRE_NewIJVector( MPI_Comm comm, HYPRE_IJVector *in_vector_ptr, 
          int global_n)

{
   int ierr=0;

   hypre_IJVector    *vector;

   vector = hypre_CTAlloc(hypre_IJVector, 1);

   hypre_IJVectorContext(vector) = comm;
   hypre_IJVectorN(vector)       = global_n;
   hypre_IJVectorLocalStorage(vector) = NULL;
   hypre_IJVectorTranslator(vector) = NULL;
   hypre_IJVectorLocalStorageType(vector) = HYPRE_UNITIALIZED;
   hypre_IJVectorInsertionSemantics(vector) = 0;
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
      hypre_IJVectorReferenceCount( vector ) --;
   
      if ( hypre_IJVectorReferenceCount( vector ) <= 0 )
      {
	/*
         if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC_VECTOR )
            ierr = hypre_FreeIJVectorPETSc( vector );
         else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS_VECTOR )
            ierr = hypre_FreeIJVectorISIS( vector );
         else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR_VECTOR )
            ierr = hypre_FreeIJVectorParCSR( vector );
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
 * HYPRE_InitializeIJVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_InitializeIJVector( HYPRE_IJVector IJvector )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC_VECTOR )
      ierr = hypre_InitializeIJVectorPETSc( vector );
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS_VECTOR )
      ierr = hypre_InitializeIJVectorISIS( vector );
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR_VECTOR )
      ierr = hypre_InitializeIJVectorParCSR( vector );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleIJVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleIJVector( HYPRE_IJVector IJvector )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   if( hypre_IJVectorN(vector) < 0 )
      return(-1);

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC_VECTOR )
      ierr = hypre_AssembleIJVectorPETSc( vector );
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS_VECTOR )
      ierr = hypre_AssembleIJVectorISIS( vector );
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR_VECTOR )
      ierr = hypre_AssembleIJVectorParCSR( vector );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributeIJVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributeIJVector( HYPRE_IJVector IJvector, int *row_starts , int *col_starts )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR_VECTOR )
      ierr = hypre_DistributeIJVectorParCSR( vector, row_starts, col_starts );
   else
      ierr = -1;

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
or a ParCSR vector from the IJ vector interface, and this choice can be controlled through this
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
@param int local_m [IN]
 local number of rows
@param int local_n [IN]
 local number of columns
HYPRE_SetIJVectorLocalStorageType needs to be called before this routine
*/

int 
HYPRE_SetIJVectorLocalSize( HYPRE_IJVector IJvector, int local_m, int local_n )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC_VECTOR )
      ierr = hypre_SetIJVectorLocalSizePETSC (vector, local_m, local_n);
   if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS_VECTOR )
      ierr = hypre_SetIJVectorLocalSizeISIS (vector, local_m, local_n);
      */
   if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR_VECTOR )
      ierr = hypre_SetIJVectorLocalSizeParCSR (vector, local_m, local_n);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorRowSizes
 *--------------------------------------------------------------------------*/

/**
* HYPRE_SetIJVectorRowSizes( HYPRE_IJVector IJvector, int *sizes); 

Not collective.
Tells "vector" how many nonzeros to expect in each row.
Knowing this quantity apriori may have a significant impact on the time needed for the Assemble phase,
and this option should always be utilized if the information is available.It is most useful in conjunction
with the next function.
@return integer error code
@param HYPRE_IJVector &vector [IN] 
the vector to be operated on.
@param int *sizes [IN]
a vector of length = local_m giving the estimated sizes for the diagonal parts of
all local_m rows, in order from lowest globally numbered local row to highest.
*/

int 
HYPRE_SetIJVectorRowSizes( HYPRE_IJVector IJvector, int *sizes )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC_VECTOR )
      ierr = hypre_SetIJVectorRowSizesPETSc( vector , sizes );
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS_VECTOR )
      ierr = hypre_SetIJVectorRowSizesISIS( vector , sizes );
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR_VECTOR )
      ierr = hypre_SetIJVectorRowSizesParCSR( vector , sizes );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetIJVectorTotalSize
 *--------------------------------------------------------------------------*/

/** 
Tells "vector" how many nonzeros to expect in each row. This option is preferable to using NO
"SetSizes" commands but it may lead to less efficient Assemble calls than using the two functions
above.

Not collective.
@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be initialized.
@param int size [IN]
total number of coefficients expected on this processor.

{\bf note} All sizes given in the SetIJVectorSize routines do not have to be exact, that is, it is not
an error for them to be incorrect. Incorrect values may degrade performance, however.
*/

int 
HYPRE_SetIJVectorTotalSize( HYPRE_IJVector IJvector, int size )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC_VECTOR )
      ierr = hypre_SetIJVectorTotalSizePETSc( vector , size );
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS_VECTOR )
      ierr = hypre_SetIJVectorTotalSizeISIS( vector , size );
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR_VECTOR )
      ierr = hypre_SetIJVectorTotalSizeParCSR( vector , size );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_QueryIJVectorInsertionSemantics
 *--------------------------------------------------------------------------*/

/** 
{\bf Note:} It is probably best to ignore  HYPRE_QueryIJVectorInsertionSemantics for the time being as it is really
planned as a possible future contingency and is confusing as it is now. -AJC, 6/99

There are three possible semantic "levels" in the integer parameter "row" used in Set and Add
functions, and each implementation supports one of them. This function returns a value indicating
the semantics supported by the instantiated IJVector. 
level = -1: processors may include values for "row" for any row number in the global vector.
level = 0: processors may only include values for row representing locally stored rows.
level = > 0: in addition to the above, processors may also include values for "row" representing
rows within the set of locally stored rows plus the next "level" levels of nearest neighbors, 
also known as "level sets" or "nearest neighbors".
Since level 0 is the most restrictive, it is also the easiest to implement, and the safest to use, as
all IJ matrices MUST support level 0.
In contrast, level 1 is the most general and most difficult to implement, least safe to use, and
potentially least efficient.
Levels greater than 0 represent a compromise that is appropriate for many engineering applications,
like finite element applications, where contributions to a particular vector row may be made from
more than one processor, typically in some small neighborhood around that row. 

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be initialized.
@param int *level [OUT]
level of off-processor value-setting that this implementation supports
*/

int 
HYPRE_QueryIJVectorInsertionSemantics( HYPRE_IJVector IJvector, int *level )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   *level = hypre_IJVectorInsertionSemantics(vector);

   return(ierr);

}

/*--------------------------------------------------------------------------
 * HYPRE_InsertIJVectorRows
 *--------------------------------------------------------------------------*/

/** 
Inserts a block of coefficients into an IJVector, overwriting any coefficients
in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param int m, n [IN]
the size of the block of values to be added.
@param int *rows [IN]
an integer vector of length m giving the indices in the global vector
corresponding to the rows in "values".
@param int *cols [IN]
an integer vector of length n giving the indices in the global vector
corresponding to the columns in "values".
@param double *values {IN]
The values to be inserted into the vector, stored in a dense, row-major
block of size n X 1.

*/

int 
HYPRE_InsertIJVectorRows( HYPRE_IJVector IJvector, int n,
                          int *rows, double *values)
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /*  if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC_VECTOR )
      ierr = hypre_InsertIJVectorRowsPETSc( vector, n, rows, values );
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS_VECTOR )
      ierr = hypre_InsertIJVectorRowsISIS( vector, n, rows, values );
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR_VECTOR )
      ierr = hypre_InsertIJVectorRowsParCSR( vector, n, rows, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_AddRowsToIJVector
 *--------------------------------------------------------------------------*/

/** 
Modifies the values stored in vector by adding in a block.
If there is no value already in a particular vector position, the structure is augmented
with a new entry. In the event of a collision, the corresponding values are summed.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector to be operated on.
@param int m, n [IN]
the size of the block of values to be added.
@param int *rows [IN]
an integer vector of length m giving the indices in the global vector
corresponding to the rows in "values".
@param int *cols [IN]
an integer vector of length n giving the indices in the global vector
corresponding to the columns in "values".
@param double *values {IN]
The values to be inserted into the vector, stored in a dense, row-major
block of size m X n.

*/

int 
HYPRE_AddRowsToIJVector( HYPRE_IJVector IJvector, int n,
                           int *rows, double *values)
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   /* if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PETSC_VECTOR )
      ierr = hypre_AddRowsToIJVectorPETSc( vector, n, rows, values );
   else if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_ISIS_VECTOR )
      ierr = hypre_AddRowsToIJVectorISIS( vector, n, rows, values );
   else */ if ( hypre_IJVectorLocalStorageType(vector) == HYPRE_PARCSR_VECTOR )
      ierr = hypre_AddRowsToIJVectorParCSR( vector, n, rows, values );
   else
      ierr = -1;

   return(ierr);
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

/*--------------------------------------------------------------------------
 * hypre_GetIJVectorLocalStorage
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
hypre_GetIJVectorLocalStorage( HYPRE_IJVector IJvector )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   return( hypre_IJVectorLocalStorage( vector ) );

}
