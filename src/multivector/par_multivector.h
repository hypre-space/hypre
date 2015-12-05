/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Header info for Parallel Vector data structure
 *
 *****************************************************************************/
#ifndef hypre_PAR_MULTIVECTOR_HEADER
#define hypre_PAR_MULTIVECTOR_HEADER

#include "_hypre_utilities.h"
#include "seq_Multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * hypre_ParMultiVector 
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm	      comm;
   HYPRE_Int                global_size;
   HYPRE_Int                first_index;
   HYPRE_Int      	      *partitioning;
   HYPRE_Int      	      owns_data;
   HYPRE_Int      	      owns_partitioning;
   HYPRE_Int      	      num_vectors;
   hypre_Multivector  *local_vector; 

/* using mask on "parallel" level seems to be inconvenient, so i (IL) moved it to
       "sequential" level. Also i now store it as a number of active indices and an array of 
       active indices. hypre_ParMultiVectorSetMask converts user-provided "(1,1,0,1,...)" mask 
       to the format above.
   HYPRE_Int                *mask;
*/

} hypre_ParMultivector;


/*--------------------------------------------------------------------------
 * Accessor macros for the Vector structure;
 * kinda strange macros; right hand side looks much convenient than left.....
 *--------------------------------------------------------------------------*/

#define hypre_ParMultiVectorComm(vector)             ((vector) -> comm)
#define hypre_ParMultiVectorGlobalSize(vector)       ((vector) -> global_size)
#define hypre_ParMultiVectorFirstIndex(vector)       ((vector) -> first_index)
#define hypre_ParMultiVectorPartitioning(vector)     ((vector) -> partitioning)
#define hypre_ParMultiVectorLocalVector(vector)      ((vector) -> local_vector)
#define hypre_ParMultiVectorOwnsData(vector)         ((vector) -> owns_data)
#define hypre_ParMultiVectorOwnsPartitioning(vector) ((vector) -> owns_partitioning)
#define hypre_ParMultiVectorNumVectors(vector)       ((vector) -> num_vectors)

/* field "mask" moved to "sequential" level, see structure above 
#define hypre_ParMultiVectorMask(vector)             ((vector) -> mask)
*/

/* function prototypes for working with hypre_ParMultiVector */
hypre_ParMultiVector *hypre_ParMultiVectorCreate(MPI_Comm, HYPRE_Int, HYPRE_Int *, HYPRE_Int);
HYPRE_Int hypre_ParMultiVectorDestroy(hypre_ParMultiVector *);
HYPRE_Int hypre_ParMultiVectorInitialize(hypre_ParMultiVector *);
HYPRE_Int hypre_ParMultiVectorSetDataOwner(hypre_ParMultiVector *, HYPRE_Int);
HYPRE_Int hypre_ParMultiVectorSetPartitioningOwner(hypre_ParMultiVector *, HYPRE_Int);
HYPRE_Int hypre_ParMultiVectorSetMask(hypre_ParMultiVector *, HYPRE_Int *);
HYPRE_Int hypre_ParMultiVectorSetConstantValues(hypre_ParMultiVector *, double);
HYPRE_Int hypre_ParMultiVectorSetRandomValues(hypre_ParMultiVector *, HYPRE_Int);
HYPRE_Int hypre_ParMultiVectorCopy(hypre_ParMultiVector *, hypre_ParMultiVector *);
HYPRE_Int hypre_ParMultiVectorScale(double, hypre_ParMultiVector *);
HYPRE_Int hypre_ParMultiVectorMultiScale(double *, hypre_ParMultiVector *);
HYPRE_Int hypre_ParMultiVectorAxpy(double, hypre_ParMultiVector *,
                             hypre_ParMultiVector *);

HYPRE_Int hypre_ParMultiVectorByDiag(  hypre_ParMultiVector *x,
                                 HYPRE_Int                *mask, 
                                 HYPRE_Int                n,
                                 double             *alpha,
                                 hypre_ParMultiVector *y);
                                 
HYPRE_Int hypre_ParMultiVectorInnerProd(hypre_ParMultiVector *, 
                                      hypre_ParMultiVector *, double *, double *);
HYPRE_Int hypre_ParMultiVectorInnerProdDiag(hypre_ParMultiVector *, 
                                      hypre_ParMultiVector *, double *, double *);
HYPRE_Int
hypre_ParMultiVectorCopyWithoutMask(hypre_ParMultiVector *x, hypre_ParMultiVector *y);
HYPRE_Int
hypre_ParMultiVectorByMatrix(hypre_ParMultiVector *x, HYPRE_Int rGHeight, HYPRE_Int rHeight, 
                              HYPRE_Int rWidth, double* rVal, hypre_ParMultiVector * y);
HYPRE_Int
hypre_ParMultiVectorXapy(hypre_ParMultiVector *x, HYPRE_Int rGHeight, HYPRE_Int rHeight, 
                              HYPRE_Int rWidth, double* rVal, hypre_ParMultiVector * y);
                                      
HYPRE_Int
hypre_ParMultiVectorEval(void (*f)( void*, void*, void* ), void* par,
                           hypre_ParMultiVector * x, hypre_ParMultiVector * y);

/* to be replaced by better implementation when format for multivector files established */
hypre_ParMultiVector * hypre_ParMultiVectorTempRead(MPI_Comm comm, const char *file_name);
HYPRE_Int hypre_ParMultiVectorTempPrint(hypre_ParMultiVector *vector, const char *file_name);

#ifdef __cplusplus
}
#endif

#endif   /* hypre_PAR_MULTIVECTOR_HEADER */
