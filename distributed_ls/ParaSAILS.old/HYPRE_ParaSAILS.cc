/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include <assert.h>

#include "HYPRE_ParaSAILS.h"
#include "hypre_ParaSAILS_i.h"

#include "parcsr_matrix_vector.h"
#include "HYPRE_IJ_mv.h"


/*--------------------------------------------------------------------------
 * HYPRE_ParaSAILS_New
 *--------------------------------------------------------------------------*/

HYPRE_ParaSAILS ParaSAILS_New(
  MPI_Comm comm,
  HYPRE_DistributedMatrix matrix)
{
    hypre_ParaSAILS *solver;

    solver = (hypre_ParaSAILS *) hypre_CTAlloc(hypre_ParaSAILS, 1);

    solver->obj = new ParaSAILS(matrix);

    solver->obj->set_prune_alg(2);
    solver->obj->set_lfil(3);
    solver->obj->set_level(3);
    solver->obj->set_thresh(5.0);
    solver->obj->set_dump(0);

    solver->comm = comm;
    solver->matrix = matrix;

    return (HYPRE_ParaSAILS) solver;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSAILS_Free
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSAILS_Free(
  HYPRE_ParaSAILS in_ptr)
{

    hypre_ParaSAILS *solver = (hypre_ParaSAILS *) in_ptr;

    // UNDONE destroy communication package

    hypre_TFree(solver);

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSAILS_Init
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSAILS_Init( 
  HYPRE_ParaSAILS in_ptr)
{
    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSAILS_SetMat
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSAILS_SetMat(
  HYPRE_ParaSAILS in_ptr,
  HYPRE_DistributedMatrix matrix)
{
    hypre_ParaSAILS *solver = (hypre_ParaSAILS *) in_ptr;

    solver->matrix = matrix;

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSAILS_GetMat
 *--------------------------------------------------------------------------*/

HYPRE_DistributedMatrix HYPRE_ParaSAILS_GetMat( 
  HYPRE_ParaSAILS in_ptr)
{
    hypre_ParaSAILS *solver = (hypre_ParaSAILS *) in_ptr;

    return solver->matrix;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSAILS_Setup
 *--------------------------------------------------------------------------*/

//void *hypre_GetIJMatrixLocalStorage((HYPRE_IJMatrix IJMatrix));

int HYPRE_ParaSAILS_Setup(
  HYPRE_ParaSAILS in_ptr)
{
    int ierr;

    hypre_ParaSAILS *solver = (hypre_ParaSAILS *) in_ptr;

    solver->obj->calculate();

    ierr = HYPRE_IJMatrixAssemble(solver->obj->M);
    assert(!ierr);

    // Extract the underlying ParCSR matrix from the IJ matrix
    solver->par_matrix = (hypre_ParCSRMatrix *) 
      HYPRE_IJMatrixGetLocalStorage(solver->obj->M);

    // Mat-Vec preprocessing
    hypre_MatvecCommPkgCreate(solver->par_matrix);

    return 0;
}


/*--------------------------------------------------------------------------
 * HYPRE_ParaSAILS_Solve
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSAILS_Solve(
  HYPRE_ParaSAILS in_ptr,
  hypre_ParVector *x, 
  hypre_ParVector *b)
{
    hypre_ParaSAILS *solver = (hypre_ParaSAILS *) in_ptr;

    hypre_ParCSRMatrixMatvec(1.0, solver->par_matrix, b, 1.0, x);

    return 0;
}

