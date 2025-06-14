/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_KRYLOV_SOLVER_PTRS_HEADER
#define hypre_KRYLOV_SOLVER_PTRS_HEADER

typedef void *     (*hypre_KrylovPtrToCAlloc)            (size_t count, size_t elt_size,
                                                          HYPRE_MemoryLocation location);
typedef HYPRE_Int  (*hypre_KrylovPtrToFree)              (void *ptr);
typedef HYPRE_Int  (*hypre_KrylovPtrToCommInfo)          (void *A, HYPRE_Int *my_id,
                                                          HYPRE_Int *num_procs);
typedef void *     (*hypre_KrylovPtrToCreateVector)      (void *vector);
typedef void *     (*hypre_KrylovPtrToCreateVectorArray) (HYPRE_Int size, void *vectors);
typedef HYPRE_Int  (*hypre_KrylovPtrToDestroyVector)     (void *vector);
typedef void *     (*hypre_KrylovPtrToMatvecCreate)      (void *A, void *x);
typedef HYPRE_Int  (*hypre_KrylovPtrToMatvec)            (void *matvec_data, HYPRE_Complex alpha,
                                                          void *A, void *x,
                                                          HYPRE_Complex beta, void *y);
typedef HYPRE_Int  (*hypre_KrylovPtrToMatvecT)           (void *matvec_data, HYPRE_Complex alpha,
                                                          void *A, void *x,
                                                          HYPRE_Complex beta, void *y);
typedef HYPRE_Int  (*hypre_KrylovPtrToMatvecDestroy)     (void *matvec_data);
typedef HYPRE_Real (*hypre_KrylovPtrToInnerProd)         (void *x, void *y);
typedef HYPRE_Int  (*hypre_KrylovPtrToCopyVector)        (void *x, void *y);
typedef HYPRE_Int  (*hypre_KrylovPtrToClearVector)       (void *x);
typedef HYPRE_Int  (*hypre_KrylovPtrToScaleVector)       (HYPRE_Complex alpha, void *x);
typedef HYPRE_Int  (*hypre_KrylovPtrToAxpy)              (HYPRE_Complex alpha, void *x, void *y);
typedef HYPRE_Int  (*hypre_KrylovPtrToPrecondSetup)      (void *vdata, void *A, void *b, void *x);
typedef HYPRE_Int  (*hypre_KrylovPtrToPrecond)           (void *vdata, void *A, void *b, void *x);
typedef HYPRE_Int  (*hypre_KrylovPtrToPrecondT)          (void *vdata, void *A, void *b, void *x);
typedef HYPRE_Int  (*hypre_KrylovPtrToMassInnerProd)     (void *x, void **p, HYPRE_Int k,
                                                          HYPRE_Int unroll, void *result);
typedef HYPRE_Int  (*hypre_KrylovPtrToMassDotpTwo)       (void *x, void *y, void **p, HYPRE_Int k,
                                                          HYPRE_Int unroll, void *result_x,
                                                          void *result_y);
typedef HYPRE_Int  (*hypre_KrylovPtrToMassAxpy)          (HYPRE_Complex *alpha, void **x,
                                                          void *y, HYPRE_Int k, HYPRE_Int unroll);
typedef HYPRE_Int  (*hypre_KrylovPtrToModifyPC)          (void *precond_data, HYPRE_Int iteration,
                                                          HYPRE_Real rel_residual_norm );

#endif
