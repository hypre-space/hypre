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
                          int jlower,
                          int jupper,
                          HYPRE_IJVector *vector )
{
   hypre_IJVector *vec;
   int num_procs, my_id, *partitioning;
   int ierr, i, i2;
   int *info;
   int *recv_buf;

   vec = hypre_CTAlloc(hypre_IJVector, 1);
   
   if (!vec)
   {  
      printf("Out of memory -- HYPRE_IJVectorCreate\n");
      exit(1);
   }

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

   info = hypre_CTAlloc(int,2);
   recv_buf = hypre_CTAlloc(int, 2*num_procs);
   partitioning = hypre_CTAlloc(int, num_procs+1);

   info[0] = jlower;
   info[1] = jupper;

   ierr = MPI_Allgather(info, 2, MPI_INT, recv_buf, 2, MPI_INT, comm);

   partitioning[0] = recv_buf[0];
   for (i=0; i < num_procs-1; i++)
   {
      i2 = i+i;
      if (recv_buf[i2+1] != (recv_buf[i2+2]-1))
      {
         printf("Inconsistent partitioning -- HYPRE_IJVectorCreate\n");  
	 ierr = -9;
	 return ierr;
      }
      else
	 partitioning[i+1] = recv_buf[i2+2];
   }
   i2 = (num_procs-1)*2;
   partitioning[num_procs] = recv_buf[i2+1]+1;

   hypre_TFree(info);
   hypre_TFree(recv_buf);

   hypre_IJVectorComm(vec)         = comm;
   hypre_IJVectorPartitioning(vec) = partitioning;
   hypre_IJVectorObject(vec)       = NULL;
   hypre_IJVectorObjectType(vec)   = HYPRE_UNITIALIZED;

   *vector = (HYPRE_IJVector) vec;
  
   return ierr; 
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorDestroy( HYPRE_IJVector vector )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;
   int ierr = 0;

   if (!vec)
   {
     printf("Vector variable is NULL -- HYPRE_IJVectorDestroy\n");
     exit(1);
   } 

   if (hypre_IJVectorPartitioning(vec))
      hypre_TFree(hypre_IJVectorPartitioning(vec));

   /* if ( hypre_IJVectorObjectType(vec) == HYPRE_PETSC )

      ierr = hypre_IJVectorDestroyPETSc(vec) ;

   else if ( hypre_IJVectorObjectType(vec) == HYPRE_ISIS )

      ierr = hypre_IJVectorDestroyISIS(vec) ;

   else */

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )

      ierr = hypre_IJVectorDestroyPar(vec) ;

   else
   {
      printf("Unrecognized object type -- HYPRE_IJVectorDestroy\n");
      exit(1);
   }

   hypre_TFree(vec);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorInitialize( HYPRE_IJVector vector )
{
   int ierr = 0;
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
     printf("Vector variable is NULL -- HYPRE_IJVectorInitialize\n");
     exit(1);
   } 

   /* if ( hypre_IJVectorObjectType(vec) == HYPRE_PETSC )

      return( hypre_IJVectorInitializePETSc(vec) );

   else if ( hypre_IJVectorObjectType(vec) == HYPRE_ISIS )

      return( hypre_IJVectorInitializeISIS(vec) );

   else */

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
   {
      if (!hypre_IJVectorObject(vec))
	 ierr += hypre_IJVectorCreatePar(vec,
                          hypre_IJVectorPartitioning(vec));

      ierr += hypre_IJVectorInitializePar(vec);

      return ierr;
   }
   else
   {
      printf("Unrecognized object type -- HYPRE_IJVectorInitialize\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorSetValues( HYPRE_IJVector  vector,
                         int             nvalues,
                         const int      *indices,
                         const double   *values   )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
     printf("Variable vec is NULL -- HYPRE_IJVectorSetValues\n");
     exit(1);
   } 

   /*  if ( hypre_IJVectorObjectType(vec) == HYPRE_PETSC )

      return( hypre_IJVectorSetValuesPETSc(vec, nvalues, indices, values) );

   else if ( hypre_IJVectorObjectType(vec) == HYPRE_ISIS )

      return( hypre_IJVectorSetValuesISIS(vec, nvalues, indices, values) );

   else */

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )

      return( hypre_IJVectorSetValuesPar(vec, nvalues, indices, values) );

   else
   {
      printf("Unrecognized object type -- HYPRE_IJVectorSetValues\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAddToValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorAddToValues( HYPRE_IJVector  vector,
                           int             nvalues,
                           const int      *indices,
                           const double   *values      )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
     printf("Variable vec is NULL -- HYPRE_IJVectorAddToValues\n");
     exit(1);
   } 

   /* if ( hypre_IJVectorObjectType(vec) == HYPRE_PETSC )

      return( hypre_IJVectorAddToValuesPETSc(vec, nvalues, indices, values) );

   else if ( hypre_IJVectorObjectType(vec) == HYPRE_ISIS )

      return( hypre_IJVectorAddToValuesISIS(vec, nvalues, indices, values) );

   else */ if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )

      return( hypre_IJVectorAddToValuesPar(vec, nvalues, indices, values) );

   else
   {
      printf("Unrecognized object type -- HYPRE_IJVectorAddToValues\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorAssemble( HYPRE_IJVector  vector )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
     printf("Variable vec is NULL -- HYPRE_IJVectorAssemble\n");
     exit(1);
   } 

   /* if ( hypre_IJVectorObjectType(vec) == HYPRE_PETSC )

      return( hypre_IJVectorAssemblePETSc(vec) );

   else if ( hypre_IJVectorObjectType(vec) == HYPRE_ISIS )

      return( hypre_IJVectorAssembleISIS(vec) );

   else */ if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )

      return( hypre_IJVectorAssemblePar(vec) );

   else 
   {
      printf("Unrecognized object type -- HYPRE_IJVectorAssemble\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorGetValues( HYPRE_IJVector  vector,
                         int             nvalues,
                         const int      *indices,
                         double         *values   )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
     printf("Variable vec is NULL -- HYPRE_IJVectorGetValues\n");
     exit(1);
   } 

   /* if ( hypre_IJVectorObjectType(vec) == HYPRE_PETSC )

      return( hypre_GetIJVectorPETScLocalComponents(vec, nvalues, indices, values) );

   else if ( hypre_IJVectorObjectType(vec) == HYPRE_ISIS )

      return( hypre_IJVectorGetValuesISIS(vec, nvalues, indices, values) );

   else */ if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )

      return( hypre_IJVectorGetValuesPar(vec, nvalues, indices, values) );

   else
   {
      printf("Unrecognized object type -- HYPRE_IJVectorGetValues\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetObjectType
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJVectorSetObjectType( HYPRE_IJVector vector, int type )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
     printf("Variable vec is NULL -- HYPRE_IJVectorSetObjectType\n");
     exit(1);
   } 

   hypre_IJVectorObjectType(vec) = type;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObjectType
 *--------------------------------------------------------------------------*/

int
HYPRE_IJVectorGetObjectType( HYPRE_IJVector vector, int *type )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
     printf("Variable vec is NULL -- HYPRE_IJVectorGetObjectType\n");
     exit(1);
   } 

   *type = hypre_IJVectorObjectType(vec);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObject
 *--------------------------------------------------------------------------*/

int
HYPRE_IJVectorGetObject( HYPRE_IJVector vector, void **object )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
     printf("Variable vec is NULL -- HYPRE_IJVectorGetObject\n");
     exit(1);
   } 

   *object = hypre_IJVectorObject(vec);

   if (*object != NULL)
     return 0; 

   return -1;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorRead
 *--------------------------------------------------------------------------*/

int
HYPRE_IJVectorRead( char           *filename,
                    MPI_Comm        comm,
                    int             object_type,
                    HYPRE_IJVector *vector )
{
   int ierr = 0;
   hypre_IJVector *vec;

   /* if ( object_type == HYPRE_PETSC )
      ierr =  hypre_IJVectorReadPETSc( comm, filename, &vec );
   else if ( object_type == HYPRE_ISIS )
      ierr = hypre_IJVectorReadISIS( comm, filename, &vec );
   else */

   if ( object_type == HYPRE_PARCSR )
      ierr = hypre_IJVectorReadPar( comm, filename, &vec );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJVectorRead\n");
      exit(1);
   }

   hypre_IJVectorObjectType(vec) = object_type;

   *vector =  (HYPRE_IJVector) vec;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorPrint
 *--------------------------------------------------------------------------*/

int
HYPRE_IJVectorPrint( HYPRE_IJVector vector, char *filename )
{
   int ierr = 0;
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (!vec)
   {
      printf("Variable vec is NULL -- HYPRE_IJVectorPrint\n");
      exit(1);
   } 

   /* if ( hypre_IJVectorObjectType(vec) == HYPRE_PETSC )
      ierr =  hypre_IJVectorPrintPETSc( vec, filename );
   else if ( hypre_IJVectorObjectType(vec) == HYPRE_ISIS )
      ierr = hypre_IJVectorPrintISIS( vec, filename );
   else */

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )
      ierr = hypre_IJVectorPrintPar( vec, filename );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJVectorPrint\n");
      exit(1);
   }

   return ierr;
}
