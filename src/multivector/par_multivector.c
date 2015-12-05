/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.10 $
 ***********************************************************************EHEADER*/





#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "par_multivector.h"
#include "seq_multivector.h"

#include "_hypre_utilities.h"

/* for temporary implementation of multivectorRead, multivectorPrint */
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

hypre_ParMultiVector  *
hypre_ParMultiVectorCreate(MPI_Comm comm, HYPRE_Int global_size, HYPRE_Int *partitioning,
                           HYPRE_Int num_vectors)
{
   hypre_ParMultiVector *vector;
   HYPRE_Int num_procs, my_id;
   
   vector = hypre_CTAlloc(hypre_ParMultiVector, 1);
   
   hypre_MPI_Comm_rank(comm, &my_id);
   
   if (! partitioning)
   {
      hypre_MPI_Comm_size(comm, &num_procs);
      hypre_GeneratePartitioning(global_size, num_procs, &partitioning);
   }
   
   hypre_ParMultiVectorComm(vector) = comm;
   hypre_ParMultiVectorGlobalSize(vector) = global_size;
   hypre_ParMultiVectorPartitioning(vector) = partitioning;
   hypre_ParMultiVectorNumVectors(vector) = num_vectors;
   
   hypre_ParMultiVectorLocalVector(vector) = 
         hypre_SeqMultivectorCreate((partitioning[my_id+1]-partitioning[my_id]), num_vectors);
   
   hypre_ParMultiVectorFirstIndex(vector) = partitioning[my_id];
  
  /* we set these 2 defaults exactly as in par_vector.c, although it's questionable */
   hypre_ParMultiVectorOwnsData(vector) = 1;
   hypre_ParMultiVectorOwnsPartitioning(vector) = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParMultiVectorDestroy( hypre_ParMultiVector *pm_vector )
{
   if (NULL!=pm_vector)
   {
      if ( hypre_ParMultiVectorOwnsData(pm_vector) )
         hypre_SeqMultivectorDestroy(hypre_ParMultiVectorLocalVector(pm_vector));
      
      if ( hypre_ParMultiVectorOwnsPartitioning(pm_vector) )
         hypre_TFree(hypre_ParMultiVectorPartitioning(pm_vector));
      
      hypre_TFree(pm_vector);
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParMultiVectorInitialize( hypre_ParMultiVector *pm_vector )
{
   HYPRE_Int  ierr;

   ierr = hypre_SeqMultivectorInitialize(
                               hypre_ParMultiVectorLocalVector(pm_vector));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParMultiVectorSetDataOwner( hypre_ParMultiVector *pm_vector,
                                  HYPRE_Int           owns_data   )
{
   HYPRE_Int    ierr=0;

   hypre_ParMultiVectorOwnsData(pm_vector) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorSetPartitioningOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParMultiVectorSetPartitioningOwner( hypre_ParMultiVector *pm_vector,
                             	          HYPRE_Int owns_partitioning)
{
   hypre_ParMultiVectorOwnsPartitioning(pm_vector) = owns_partitioning;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorSetMask
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_ParMultiVectorSetMask( hypre_ParMultiVector *pm_vector, HYPRE_Int *mask)
{

   return hypre_SeqMultivectorSetMask(pm_vector->local_vector, mask);
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParMultiVectorSetConstantValues( hypre_ParMultiVector *v,
                                       double               value )
{
   hypre_Multivector *v_local = hypre_ParMultiVectorLocalVector(v);
           
   return hypre_SeqMultivectorSetConstantValues(v_local,value);
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorSetRandomValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParMultiVectorSetRandomValues( hypre_ParMultiVector *v, HYPRE_Int  seed)
{
   HYPRE_Int my_id;
   hypre_Multivector *v_local = hypre_ParMultiVectorLocalVector(v);

   MPI_Comm 	comm = hypre_ParMultiVectorComm(v);
   hypre_MPI_Comm_rank(comm,&my_id); 

   seed *= (my_id+1);
           
   return hypre_SeqMultivectorSetRandomValues(v_local,seed);
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParMultiVectorCopy(hypre_ParMultiVector *x, hypre_ParMultiVector *y)
{
   hypre_Multivector *x_local = hypre_ParMultiVectorLocalVector(x);
   hypre_Multivector *y_local = hypre_ParMultiVectorLocalVector(y);  
   
   return hypre_SeqMultivectorCopy(x_local, y_local);
}


HYPRE_Int
hypre_ParMultiVectorCopyWithoutMask(hypre_ParMultiVector *x, hypre_ParMultiVector *y)
{
   return hypre_SeqMultivectorCopyWithoutMask(x->local_vector, y->local_vector);
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParMultiVectorScale(double alpha, hypre_ParMultiVector *y)
{
   return 1 ; /* hypre_SeqMultivectorScale( alpha, y_local, NULL); */
}


/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorMultiScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParMultiVectorMultiScale(double *alpha, hypre_ParMultiVector *y)
{
   return 1; /* hypre_SeqMultivectorMultiScale(alpha, y_local, NULL); */
}



/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParMultiVectorAxpy(double alpha, hypre_ParMultiVector *x,
                         hypre_ParMultiVector *y)
{
   hypre_Multivector *x_local = hypre_ParMultiVectorLocalVector(x);
   hypre_Multivector *y_local = hypre_ParMultiVectorLocalVector(y);
           
   return hypre_SeqMultivectorAxpy( alpha, x_local, y_local);
}


/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorByDiag
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParMultiVectorByDiag(hypre_ParMultiVector *x, HYPRE_Int *mask, HYPRE_Int n,
                           double *alpha, hypre_ParMultiVector *y)
{
   return hypre_SeqMultivectorByDiag(x->local_vector, mask, n, alpha, 
                                      y->local_vector);
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParMultiVectorInnerProd(hypre_ParMultiVector *x, hypre_ParMultiVector *y,
                              double *results, double *workspace )
{
   MPI_Comm           comm;
   HYPRE_Int                count;
   HYPRE_Int                ierr;
/*
 *    HYPRE_Int                myid;
 *    HYPRE_Int                i
 */

   /* assuming "results" and "workspace" are arrays of size ("n_active_x" by "n_active_y")
      n_active_x is the number of active vectors in multivector x 
      the product "x^T * y" will be stored in "results" column-wise; workspace will be used for
      computation of local matrices; maybe hypre_MPI_IN_PLACE functionality will be added later */
      
   hypre_SeqMultivectorInnerProd(x->local_vector, y->local_vector, workspace);
   
   comm = x->comm;
   count = (x->local_vector->num_active_vectors) * 
           (y->local_vector->num_active_vectors);
   
   ierr = hypre_MPI_Allreduce(workspace, results, count, hypre_MPI_DOUBLE, hypre_MPI_SUM, comm);
   hypre_assert (ierr == hypre_MPI_SUCCESS);

/* debug */

/*
 *    hypre_MPI_Comm_rank(comm, &myid);
 *    if (myid==0)
 *       for (i=0; i<count; i++)
 *          hypre_printf("%22.14e\n",results[i])
 */

/* ------------ */

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParMultiVectorInnerProdDiag
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParMultiVectorInnerProdDiag(hypre_ParMultiVector *x, hypre_ParMultiVector *y,
                                  double *diagResults, double *workspace )
{
   HYPRE_Int   count;
   HYPRE_Int   ierr;
   
   hypre_SeqMultivectorInnerProdDiag(x->local_vector, y->local_vector, workspace);
   
   count = x->local_vector->num_active_vectors;
   ierr = hypre_MPI_Allreduce(workspace, diagResults, count, hypre_MPI_DOUBLE, hypre_MPI_SUM, x->comm);
   hypre_assert (ierr == hypre_MPI_SUCCESS);
   
   return 0;
}

HYPRE_Int
hypre_ParMultiVectorByMatrix(hypre_ParMultiVector *x, HYPRE_Int rGHeight, HYPRE_Int rHeight, 
                             HYPRE_Int rWidth, double* rVal, hypre_ParMultiVector * y)
{
   return hypre_SeqMultivectorByMatrix(x->local_vector, rGHeight, rHeight,
                                        rWidth, rVal, y->local_vector);
}

HYPRE_Int
hypre_ParMultiVectorXapy(hypre_ParMultiVector *x, HYPRE_Int rGHeight, HYPRE_Int rHeight, 
                         HYPRE_Int rWidth, double* rVal, hypre_ParMultiVector * y)
{
   return hypre_SeqMultivectorXapy(x->local_vector, rGHeight, rHeight,
                                    rWidth, rVal, y->local_vector);
}

/* temporary function; allows to do "matvec" and preconditioner in 
   vector-by-vector fashion */
HYPRE_Int
hypre_ParMultiVectorEval(void (*f)( void*, void*, void* ), void* par,
                           hypre_ParMultiVector * x, hypre_ParMultiVector * y)
{
   hypre_ParVector  *temp_x, *temp_y;
   HYPRE_Int i;
   HYPRE_Int num_active_vectors;
   HYPRE_Int *x_active_indices, *y_active_indices;
   double * x_data, *y_data;
   HYPRE_Int size;
   
   hypre_assert(x->local_vector->num_active_vectors == y->local_vector->num_active_vectors);
   hypre_assert(x->local_vector->size == y->local_vector->size);
   
   temp_x=hypre_ParVectorCreate(x->comm,x->global_size, x->partitioning);
   hypre_assert(temp_x!=NULL);
   temp_x->owns_partitioning=0;
   temp_x->local_vector->owns_data=0;
   temp_x->local_vector->vecstride = temp_x->local_vector->size;
   temp_x->local_vector->idxstride = 1;
/* no initialization for temp_x needed! */
  
   temp_y=hypre_ParVectorCreate(y->comm,y->global_size, y->partitioning);
   hypre_assert(temp_y!=NULL);
   temp_y->owns_partitioning=0;
   temp_y->local_vector->owns_data=0;
   temp_y->local_vector->vecstride = temp_y->local_vector->size;
   temp_y->local_vector->idxstride = 1;
/* no initialization for temp_y needed! */     

   num_active_vectors = x->local_vector->num_active_vectors;
   x_active_indices = x->local_vector->active_indices;
   y_active_indices = y->local_vector->active_indices;
   x_data = x->local_vector->data;
   y_data = y->local_vector->data;
   size = x->local_vector->size;
   
   for ( i=0; i<num_active_vectors; i++ ) 
   {
      temp_x->local_vector->data = x_data + x_active_indices[i]*size;
      temp_y->local_vector->data = y_data + y_active_indices[i]*size;
   
      /*** here i make an assumption that "f" will treat temp_x and temp_y like 
            "hypre_ParVector *" variables ***/

      f( par, temp_x, temp_y );
   }
   
   hypre_ParVectorDestroy(temp_x);
   hypre_ParVectorDestroy(temp_y); 
   /* 2 lines above won't free data or partitioning */

   return 0;
}

hypre_ParMultiVector * 
hypre_ParMultiVectorTempRead(MPI_Comm comm, const char *fileName)
        /* ***** temporary implementation ****** */
{
   HYPRE_Int i, n, id;
   double * dest;
   double * src;
   HYPRE_Int count;
   HYPRE_Int retcode;
   char temp_string[128];
   hypre_ParMultiVector * x;
   hypre_ParVector * temp_vec;
     
   /* calculate the number of files */
   hypre_MPI_Comm_rank( comm, &id );
   n = 0;
   do {
     hypre_sprintf( temp_string, "test -f %s.%d.%d", fileName, n, id ); 
     if (!(retcode=system(temp_string))) /* zero retcode mean file exists */
       n++;
   } while (!retcode);

   if ( n == 0 ) return NULL;

   /* now read the first vector using hypre_ParVectorRead into temp_vec */

   hypre_sprintf(temp_string,"%s.%d",fileName,0);
   temp_vec = hypre_ParVectorRead(comm, temp_string);

   /* this vector WON'T own partitioning */
   hypre_ParVectorSetPartitioningOwner(temp_vec,0);

   /* now create multivector using temp_vec as a sample */

   x = hypre_ParMultiVectorCreate(hypre_ParVectorComm(temp_vec),
        hypre_ParVectorGlobalSize(temp_vec),hypre_ParVectorPartitioning(temp_vec),n);

   /* this vector WILL own the partitioning */
   hypre_ParMultiVectorSetPartitioningOwner(x,1);
  
   hypre_ParMultiVectorInitialize(x);
  
   /* read data from first and all other vectors into "x" */

   i = 0;
   do {
   /* copy data from current vector */
      dest = x->local_vector->data + i*(x->local_vector->size);
      src = temp_vec->local_vector->data;
      count = temp_vec->local_vector->size;
   
      memcpy(dest,src, count*sizeof(double));
      
   /* destroy current vector */
      hypre_ParVectorDestroy(temp_vec);

   /* read the data to new current vector, if there are more vectors to read */   
      if (i<n-1)
      {
         hypre_sprintf(temp_string,"%s.%d",fileName,i+1);
         temp_vec = hypre_ParVectorRead(comm, temp_string);
         
      }
   } while (++i<n);

   return x; 
}

HYPRE_Int 
hypre_ParMultiVectorTempPrint(hypre_ParMultiVector *vector, const char *fileName)
{
   HYPRE_Int i, ierr;
   char fullName[128];
   hypre_ParVector * temp_vec;
  
   hypre_assert( vector != NULL );
  
   temp_vec=hypre_ParVectorCreate(vector->comm,vector->global_size, vector->partitioning);
   hypre_assert(temp_vec!=NULL);
   temp_vec->owns_partitioning=0;
   temp_vec->local_vector->owns_data=0;
  
   /* no initialization for temp_vec needed! */
  
   ierr = 0;
   for ( i = 0; i < vector->local_vector->num_vectors; i++ ) 
   {
     hypre_sprintf( fullName, "%s.%d", fileName, i ); 
   
     temp_vec->local_vector->data=vector->local_vector->data + i * 
                                     vector->local_vector->size;
   
     ierr = ierr || hypre_ParVectorPrint(temp_vec, fullName);
   }
   
   ierr = ierr || hypre_ParVectorDestroy(temp_vec); 
   /* line above won't free data or partitioning */
 
   return ierr;
}

