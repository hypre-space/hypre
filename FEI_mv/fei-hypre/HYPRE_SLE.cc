/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/


#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <fstream.h>
#include <math.h>

#include "mpi.h"

// include matrix-vector files from ISIS++

#include "other/basicTypes.h"
#include "RealArray.h"
#include "IntArray.h"
#include "GlobalIDArray.h"
#include "CommInfo.h"
#include "Map.h"
#include "Vector.h"
#include "Matrix.h"


// include the Hypre package header here

#include "HYPRE_IJ_mv.h"
#include "HYPRE.h" // needed for HYPRE_PARCSR_MATRIX

#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"

// includes needed in order to be able to include BASE_SLE.h

#include "other/basicTypes.h"
#include "fei.h"
#include "src/CommBufferDouble.h"
#include "src/CommBufferInt.h"
#include "src/NodePackets.h"
#include "src/BCRecord.h"
#include "src/FieldRecord.h"
#include "src/BlockRecord.h"
#include "src/MultConstRecord.h"
#include "src/PenConstRecord.h"
#include "src/SimpleList.h"
#include "src/NodeRecord.h"
#include "src/SharedNodeRecord.h"
#include "src/SharedNodeBuffer.h"
#include "src/ExternNodeRecord.h"
#include "src/SLE_utils.h"

#include "src/BASE_SLE.h"

#include "HYPRE_SLE.h"


//------------------------------------------------------------------------------
HYPRE_SLE::HYPRE_SLE(MPI_Comm PASSED_COMM_WORLD, int masterRank) : 
  BASE_SLE(PASSED_COMM_WORLD, masterRank)
{
    comm = PASSED_COMM_WORLD;

    pcg_solver = NULL;
    pcg_precond = NULL;
}

//------------------------------------------------------------------------------
HYPRE_SLE::~HYPRE_SLE() {
//
//  Destructor function. Free allocated memory, etc.
//
    deleteLinearAlgebraCore();
}


//------------------------------------------------------------------------------
void HYPRE_SLE::selectSolver(char *name)
{
    if (!strcmp(name, "pcg")) 
    {
      HYPRE_ParCSRPCGInitialize(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRPCGSetMaxIter(pcg_solver, 500);
      HYPRE_ParCSRPCGSetTol(pcg_solver, 1.e-10);
      HYPRE_ParCSRPCGSetTwoNorm(pcg_solver, 1);
      HYPRE_ParCSRPCGSetRelChange(pcg_solver, 0);
      HYPRE_ParCSRPCGSetLogging(pcg_solver, 1);
    }
    else // if (!strcmp(name, "gmres")) 
    {
      HYPRE_ParCSRGMRESInitialize(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRGMRESSetKDim(pcg_solver, 50);
      HYPRE_ParCSRGMRESSetMaxIter(pcg_solver, 100);
      HYPRE_ParCSRGMRESSetTol(pcg_solver, 1.e-10);
      HYPRE_ParCSRGMRESSetLogging(pcg_solver, 1);
    }
}

//------------------------------------------------------------------------------
void HYPRE_SLE::selectPreconditioner(char *name)
{
    // selectSolver must be called first
    assert(pcg_solver != NULL);

    if (!strcmp(name, "identity")) 
    {
    }
    else if (!strcmp(name, "boomeramg")) 
    {
         pcg_precond = HYPRE_ParAMGInitialize();
#if 0
         HYPRE_ParAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_ParAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_ParAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_ParAMGSetLogging(pcg_precond, ioutdat, "driver.out.log");
         HYPRE_ParAMGSetMaxIter(pcg_precond, 1);
         HYPRE_ParAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_ParAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_ParAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_ParAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_ParAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_ParAMGSetMaxLevels(pcg_precond, max_levels);
#endif

/* prototype for this function needs to be declared properly
         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup,
                                   pcg_precond);
*/

    }
    else if (!strcmp(name, "parasails")) 
    {
    }
    else if (!strcmp(name, "pilut")) 
    {
    }
    else // if (!strcmp(name, "diagonal")) 
    {
         pcg_precond = NULL;

/* prototype for this function needs to be declared properly
         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
*/

    }
}

//------------------------------------------------------------------------------
// This function is called by the constructor, just initializes
// the pointers and other variables associated with the linear
// algebra core to NULL or appropriate initial values.

void HYPRE_SLE::initLinearAlgebraCore()
{
    A = (HYPRE_IJMatrix) NULL;
    x = (HYPRE_IJVector) NULL;
    b = (HYPRE_IJVector) NULL;
}

//------------------------------------------------------------------------------
//This is a destructor-type function.
//This function deletes allocated memory associated with
//the linear algebra core objects/data structures.

void HYPRE_SLE::deleteLinearAlgebraCore()
{

    HYPRE_FreeIJMatrix(A);
    HYPRE_FreeIJVector(x);
    HYPRE_FreeIJVector(b);
}

//------------------------------------------------------------------------------
//This function is where we establish the structures/objects associated
//with the linear algebra library. i.e., do initial allocations, etc.

void HYPRE_SLE::createLinearAlgebraCore(int globalNumEqns,
  int localStartRow, int localEndRow, int localStartCol, int localEndCol)
{
    int ierr;
    int *partitioning;

    first_row = localStartRow;
    last_row = localEndRow;

    ierr = HYPRE_NewIJMatrix(comm, &A, globalNumEqns, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_SetIJMatrixLocalStorageType(A, HYPRE_PARCSR_MATRIX);
    assert(!ierr);
    ierr = HYPRE_SetIJMatrixLocalSize(A, localEndRow-localStartRow+1, 
      globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_InitializeIJMatrix(A);
    assert(!ierr);

    // UNDONE: this function does not exist yet
    // HYPRE_GetIJMatrixRowPartitioning(A, &partitioning);

    ierr = HYPRE_NewIJVector(comm, &b, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_SetIJVectorLocalStorageType(b, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_SetIJVectorPartitioning(b, partitioning);
    assert(!ierr);
    ierr = HYPRE_InitializeIJVector(b);
    assert(!ierr);

    ierr = HYPRE_NewIJVector(comm, &x, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_SetIJVectorLocalStorageType(x, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_SetIJVectorPartitioning(x, partitioning);
    assert(!ierr);
    ierr = HYPRE_InitializeIJVector(x);
    assert(!ierr);
}

//------------------------------------------------------------------------------
void HYPRE_SLE::matrixConfigure(IntArray* rows)
{
    int *sizes1 = new int[last_row-first_row+1];
    int *sizes2 = new int[last_row-first_row+1];

    int i, j;

    for (i=0; i<last_row-first_row+1; i++)
    {
	int  num_indices = rows[i].size();
	int *indices = &((rows[i])[0]);

	int  num1 = 0;
	int  num2 = 0;

	for (j=0; j<num_indices; j++)
	{
	    if (first_row <= indices[j] && indices[j] <= last_row)
		num1++;
	    else
		num2++;
	}

	sizes1[i] = num1;
	sizes2[i] = num2;
    }

    HYPRE_SetIJMatrixDiagRowSizes(A, sizes1);
    HYPRE_SetIJMatrixOffDiagRowSizes(A, sizes2);

    delete sizes1;
    delete sizes2;
}

//------------------------------------------------------------------------------
// This function is needed in order to construct a new problem with the 
// same sparsity pattern.

void HYPRE_SLE::resetMatrixAndVector(double s)
{
    assert(s == 0.0);

    HYPRE_ZeroIJVectorLocalComponents(b);

    // UNDONE: set matrix to 0
    assert(0);
}

//------------------------------------------------------------------------------
void HYPRE_SLE::sumIntoRHSVector(int num, const int* indices, 
  const double* values)
{
    int i;

    int *ind = (int *) indices; // cast away const-ness

    for (i=0; i<num; i++)
	ind[i]--;               // change indices to 0-based

    HYPRE_AddToIJVectorLocalComponents(b, num, ind, NULL, values);

    for (i=0; i<num; i++)
	ind[i]++;               // change indices back to 1-based
}

//------------------------------------------------------------------------------
void HYPRE_SLE::putIntoSolnVector(int num, const int* indices,
  const double* values)
{
    int i;

    int *ind = (int *) indices; // cast away const-ness

    for (i=0; i<num; i++)
	ind[i]--;               // change indices to 0-based

    HYPRE_SetIJVectorLocalComponents(b, num, ind, NULL, values);

    for (i=0; i<num; i++)
	ind[i]++;               // change indices back to 1-based
}

//------------------------------------------------------------------------------
double HYPRE_SLE::accessSolnVector(int equation)
{
    double val;

    HYPRE_GetIJVectorLocalComponents(x, 1, &equation, NULL, &val);

    return val;
}

//------------------------------------------------------------------------------
void HYPRE_SLE::sumIntoSystemMatrix(int row, int numValues, 
  const double* values, const int* scatterIndices)
{
    int i;

    int *ind = (int *) scatterIndices; // cast away const-ness

    for (i=0; i<numValues; i++)
	ind[i]--;               // change indices to 0-based

    HYPRE_InsertIJMatrixRow(A, numValues, row, ind, (double *) values);

    for (i=0; i<numValues; i++)
	ind[i]++;               // change indices back to 1-based

}

//------------------------------------------------------------------------------
void HYPRE_SLE::enforceEssentialBC(int* globalEqn, double* alpha,
                                  double* gamma, int len) {
//
//This function must enforce an essential boundary condition on
//the equations in 'globalEqn'. This means, that the following modification
//should be made to A and b, for each globalEqn:
//
//for(all local equations i){
//   if (i==globalEqn) b[i] = gamma/alpha;
//   else b[i] -= (gamma/alpha)*A[i,globalEqn];
//}
//all of row 'globalEqn' and column 'globalEqn' in A should be zeroed,
//except for 1.0 on the diagonal.
//
}

//------------------------------------------------------------------------------
void HYPRE_SLE::enforceOtherBC(int* globalEqn, double* alpha, double* beta,
                              double* gamma, int len) {
//
//This function must enforce a natural or mixed boundary condition
//on equation 'globalEqn'. This means that the following modification should
//be made to A and b:
//
//A[globalEqn,globalEqn] += alpha/beta;
//b[globalEqn] += gamma/beta;
//
}

//------------------------------------------------------------------------------
void HYPRE_SLE::matrixLoadComplete()
{
    int ierr;
    ierr = HYPRE_AssembleIJMatrix(A);
}

//------------------------------------------------------------------------------
void HYPRE_SLE::launchSolver(int* solveStatus)
{
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;

    A_csr = (HYPRE_ParCSRMatrix) HYPRE_GetIJMatrixLocalStorage(A);
    x_csr = (HYPRE_ParVector)    HYPRE_GetIJVectorLocalStorage(x);
    b_csr = (HYPRE_ParVector)    HYPRE_GetIJVectorLocalStorage(b);

    HYPRE_ParCSRPCGSetup(pcg_solver, A_csr, b_csr, x_csr);
    HYPRE_ParCSRPCGSolve(pcg_solver, A_csr, b_csr, x_csr);

    HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_iterations);
    HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
    HYPRE_ParCSRPCGFinalize(pcg_solver);

    *solveStatus = 0; // return code
}
