/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "HYPRE.h"
#include "vector/mli_vector.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/parcsr_mv.h"
#include "util/mli_utils.h"

/******************************************************************************
 * constructor 
 *---------------------------------------------------------------------------*/

MLI_Vector::MLI_Vector( void *invec, char *in_name, MLI_Function *func_ptr )
{
   strncpy(name, in_name, 100);
   vector       = invec;
   if ( func_ptr != NULL ) destroy_func = (int (*)(void*)) func_ptr->func_;
   else                    destroy_func = NULL;
}

/******************************************************************************
 * destructor 
 *---------------------------------------------------------------------------*/

MLI_Vector::~MLI_Vector()
{
   if ( vector != NULL && destroy_func != NULL ) destroy_func((void*) vector);
   vector       = NULL;
   destroy_func = NULL;
}

/******************************************************************************
 * set vector to a constant 
 *---------------------------------------------------------------------------*/

int MLI_Vector::setConstantValue(double value)
{
   if ( strcmp( name, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::setConstantValue ERROR - type not HYPRE_ParVector\n");
      exit(1);
   }
   hypre_ParVector *vec = (hypre_ParVector *) vector;
   return (hypre_ParVectorSetConstantValues( vec, value )); 
}

/******************************************************************************
 * inner product 
 *---------------------------------------------------------------------------*/

int MLI_Vector::copy(MLI_Vector *vec2)
{
   if ( strcmp( name, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::copy ERROR - invalid type (from).\n");
      exit(1);
   }
   if ( strcmp( vec2->getName(), "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::copy ERROR - invalid type (to).\n");
      exit(1);
   }
   hypre_ParVector *hypreV1 = (hypre_ParVector *) vector;
   hypre_ParVector *hypreV2 = (hypre_ParVector *) vec2->getVector();
   hypre_ParVectorCopy( hypreV1, hypreV2 );
   return 0;
}

/******************************************************************************
 * print to a file
 *---------------------------------------------------------------------------*/

int MLI_Vector::print(char *filename)
{
   if ( strcmp( name, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::innerProduct ERROR - invalid type.\n");
      exit(1);
   }
   if ( filename == NULL ) return 1;
   hypre_ParVector *vec = (hypre_ParVector *) vector;
   hypre_ParVectorPrint( vec, filename );
   return 0;
}

/******************************************************************************
 * inner product 
 *---------------------------------------------------------------------------*/

double MLI_Vector::norm2()
{
   if ( strcmp( name, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::innerProduct ERROR - invalid type.\n");
      exit(1);
   }
   hypre_ParVector *vec = (hypre_ParVector *) vector;
   return (sqrt(hypre_ParVectorInnerProd( vec, vec )));
}

/******************************************************************************
 * clone a hypre vector 
 *---------------------------------------------------------------------------*/

MLI_Vector *MLI_Vector::clone()
{
   char            param_string[100];
   MPI_Comm        comm;
   hypre_ParVector *new_vec;
   hypre_Vector    *seq_vec;
   int             i, nlocals, global_size, *vpartition, *partitioning;
   int             mypid, nprocs;
   double          *darray;
   MLI_Function    *func_ptr;

   if ( strcmp( name, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::clone ERROR - invalid type.\n");
      exit(1);
   }
   hypre_ParVector *vec = (hypre_ParVector *) vector;
   comm = hypre_ParVectorComm(vec);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   vpartition = hypre_ParVectorPartitioning(vec);
   partitioning = hypre_CTAlloc(int,nprocs+1);
   for ( i = 0; i < nprocs+1; i++ ) partitioning[i] = vpartition[i];
   global_size = hypre_ParVectorGlobalSize(vec);
   new_vec = hypre_CTAlloc(hypre_ParVector, 1);
   hypre_ParVectorComm(new_vec) = comm;
   hypre_ParVectorGlobalSize(new_vec) = global_size;
   hypre_ParVectorFirstIndex(new_vec) = partitioning[mypid];
   hypre_ParVectorPartitioning(new_vec) = partitioning;
   hypre_ParVectorOwnsData(new_vec) = 1;
   hypre_ParVectorOwnsPartitioning(new_vec) = 1;
   nlocals = partitioning[mypid+1] - partitioning[mypid];
   seq_vec = hypre_SeqVectorCreate(nlocals);
   hypre_SeqVectorInitialize(seq_vec);
   darray = hypre_VectorData(seq_vec);
   for (i = 0; i < nlocals; i++) darray[i] = 0.0;
   hypre_ParVectorLocalVector(new_vec) = seq_vec;
   sprintf(param_string,"HYPRE_ParVector");
   func_ptr = new MLI_Function();
   MLI_Utils_HypreVectorGetDestroyFunc(func_ptr);
   MLI_Vector *mli_vec = new MLI_Vector(new_vec, param_string, func_ptr);
   delete func_ptr;
   return mli_vec;
}

