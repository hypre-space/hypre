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
#include "HYPRE.h"

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
//  Destructor function. Free allocated memory, etc.
//
HYPRE_SLE::~HYPRE_SLE()
{
    deleteLinearAlgebraCore();
}


//------------------------------------------------------------------------------
void HYPRE_SLE::selectSolver(char *name)
{
    if (!strcmp(name, "pcg")) 
    {
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
      HYPRE_ParCSRPCGSetMaxIter(pcg_solver, 500);
      HYPRE_ParCSRPCGSetTol(pcg_solver, 1.e-10);
      HYPRE_ParCSRPCGSetTwoNorm(pcg_solver, 1);
      HYPRE_ParCSRPCGSetRelChange(pcg_solver, 0);
      HYPRE_ParCSRPCGSetLogging(pcg_solver, 1);
    }
    else // if (!strcmp(name, "gmres")) 
    {
      HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &pcg_solver);
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
         // pcg_precond = HYPRE_ParAMGCreate();

         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParAMGSolve,
                                   HYPRE_ParAMGSetup,
                                   pcg_precond);

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

         HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   HYPRE_ParCSRDiagScale,
                                   HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);

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
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(x);
    HYPRE_IJVectorDestroy(b);
}

//------------------------------------------------------------------------------
//This function is where we establish the structures/objects associated
//with the linear algebra library. i.e., do initial allocations, etc.
// Rows and columns are 1-based.

void HYPRE_SLE::createLinearAlgebraCore(int globalNumEqns,
  int localStartRow, int localEndRow, int localStartCol, int localEndCol)
{
    int ierr;

    first_row = localStartRow;
    last_row = localEndRow;

    ierr = HYPRE_IJMatrixCreate(comm, &A, globalNumEqns, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalStorageType(A, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJMatrixSetLocalSize(A, localEndRow-localStartRow+1, 
      localEndRow-localStartRow+1);
    assert(!ierr);

    ierr = HYPRE_IJVectorCreate(comm, &b, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalStorageType(b, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalPartitioning(b, localStartRow-1, localEndRow);
    assert(!ierr);
    ierr = HYPRE_IJVectorInitialize(b);
    assert(!ierr);
    ierr = HYPRE_IJVectorZeroLocalComponents(b);
    assert(!ierr);

    ierr = HYPRE_IJVectorCreate(comm, &x, globalNumEqns);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalStorageType(x, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_IJVectorSetLocalPartitioning(x, localStartRow-1, localEndRow);
    assert(!ierr);
    ierr = HYPRE_IJVectorInitialize(x);
    assert(!ierr);
    ierr = HYPRE_IJVectorZeroLocalComponents(x);
    assert(!ierr);
}

//------------------------------------------------------------------------------
// Set the number of rows in the diagonal part and off diagonal part
// of the matrix, using the structure of the matrix, stored in rows.
// rows is an array that is 0-based.  first_row and last_row are 1-based.
//
void HYPRE_SLE::matrixConfigure(IntArray* rows)
{

    int *sizes1 = new int[last_row-first_row+1];
    int *sizes2 = new int[last_row-first_row+1];

    int i, j, ierr;

    for (i=0; i<last_row-first_row+1; i++)
    {
	int  num_indices = rows[first_row+i-1].size();
	int *indices = &((rows[first_row+i-1])[0]);

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

    ierr = HYPRE_IJMatrixSetDiagRowSizes(A, sizes1);
    assert(!ierr);

    ierr = HYPRE_IJMatrixSetOffDiagRowSizes(A, sizes2);
    assert(!ierr);

    ierr = HYPRE_IJMatrixInitialize(A);
    assert(!ierr);

    delete sizes1;
    delete sizes2;
}

//------------------------------------------------------------------------------
// This function is needed in order to construct a new problem with the 
// same sparsity pattern.

void HYPRE_SLE::resetMatrixAndVector(double s)
{
    assert(s == 0.0);

    HYPRE_IJVectorZeroLocalComponents(b);

    // UNDONE: set matrix to 0
    // assert(0);
}

//------------------------------------------------------------------------------
// input is 1-based, but HYPRE vectors are 0-based
//
void HYPRE_SLE::sumIntoRHSVector(int num, const int* indices, 
  const double* values)
{
    int i;
    int ierr;

    int *ind = (int *) indices; // cast away const-ness

    for (i=0; i<num; i++)
	ind[i]--;               // change indices to 0-based

    ierr = HYPRE_IJVectorAddToLocalComponents(b, num, ind, NULL, values);
    assert(ierr == 0);

    for (i=0; i<num; i++)
	ind[i]++;               // change indices back to 1-based
}

//------------------------------------------------------------------------------
// used for initializing the initial guess
//
void HYPRE_SLE::putIntoSolnVector(int num, const int* indices,
  const double* values)
{
    int i;
    int ierr;

    int *ind = (int *) indices; // cast away const-ness

    for (i=0; i<num; i++)
	ind[i]--;               // change indices to 0-based

    ierr = HYPRE_IJVectorSetLocalComponents(x, num, ind, NULL, values);
    assert(ierr == 0);

    for (i=0; i<num; i++)
	ind[i]++;               // change indices back to 1-based
}

//------------------------------------------------------------------------------
// used for getting the solution out of the solver, and into the application
//
double HYPRE_SLE::accessSolnVector(int equation)
{
    double val;
    int temp;
    int ierr;

    temp = equation-1; // construct 0-based index

    ierr = HYPRE_IJVectorGetLocalComponents(x, 1, &temp, NULL, &val);
    assert(ierr == 0);

    return val;
}

//------------------------------------------------------------------------------
void HYPRE_SLE::sumIntoSystemMatrix(int row, int numValues, 
  const double* values, const int* scatterIndices)
{
    int i;
    int ierr;

    int *ind = (int *) scatterIndices; // cast away const-ness

    for (i=0; i<numValues; i++)
	ind[i]--;               // change indices to 0-based

    ierr = HYPRE_IJMatrixAddToRow(A, numValues, row-1, ind, values);
    assert(ierr == 0);

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
    HYPRE_IJMatrixAssemble(A);

#if 0
    HYPRE_ParCSRMatrix a = (HYPRE_ParCSRMatrix) 
        HYPRE_IJMatrixGetLocalStorage(A);

    HYPRE_ParCSRMatrixPrint(a, "driver.out.a");
    exit(0);
#endif
}

//------------------------------------------------------------------------------
void HYPRE_SLE::launchSolver(int* solveStatus)
{
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;

    A_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(A);
    x_csr = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(x);
    b_csr = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(b);

    HYPRE_ParCSRPCGSetup(pcg_solver, A_csr, b_csr, x_csr);
    HYPRE_ParCSRPCGSolve(pcg_solver, A_csr, b_csr, x_csr);

    HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_iterations);
    HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
    HYPRE_ParCSRPCGDestroy(pcg_solver);

    *solveStatus = 0; // return code
}

//------------------------------------------------------------------------------

// This is a C++ test for the HYPRE FEI code in HYPRE_SLE.cc.
// It does not perform the test by calling the FEI interface functions
// which were implemented by Sandia. 
// The following is a friend function of HYPRE_SLE because it needs to call
// protected member functions of that class in order to do the test.
//
// This test program uses two processors, and sets up the following matrix.
// Rows 1-3 belong to proc 0, and rows 4-6 belong to proc 1.
// Five elements of the form [1 -1; -1 1] are summed.
// The right hand side elements are [1; -1].
//
//  1 -1  0  0  0  0      1
// -1  2 -1  0  0  0      0
//  0 -1  2 -1  0  0     -1
//  0  0 -1  2 -1  0      0
//  0  0  0 -1  2 -1      0
//  0  0  0  0 -1  1      0
//
// After the essential boundary conditions are applied, we have:

void fei_hypre_test(int argc, char *argv[])
{
    int my_rank;
    int num_procs;
    int i;
    int status;

    IntArray rows[6];
    rows[0].append(1);
    rows[0].append(2);
    rows[1].append(1);
    rows[1].append(2);
    rows[1].append(3);
    rows[2].append(2);
    rows[2].append(3);
    rows[2].append(4);
    rows[3].append(3);
    rows[3].append(4);
    rows[3].append(5);
    rows[4].append(4);
    rows[4].append(5);
    rows[4].append(6);
    rows[5].append(5);
    rows[5].append(6);

    const int ind1[] = {1, 2};
    const int ind2[] = {2, 3};
    const int ind3[] = {3, 4};
    const int ind4[] = {4, 5};
    const int ind5[] = {5, 6};

    const double val1[] = {1.0, -1.0};
    const double val2[] = {-1.0, 1.0};

    const int indg1[] = {1, 2, 3};
    const int indg2[] = {4, 5, 6};
    const double valg[] = {0.1, 0.2, 0.3};

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    assert(num_procs == 2);

    HYPRE_SLE H(MPI_COMM_WORLD, 0);

    if (my_rank == 0)
        H.createLinearAlgebraCore(6, 1, 3, 1, 6);
    else
        H.createLinearAlgebraCore(6, 4, 6, 1, 6);

    H.matrixConfigure(rows);

    switch (my_rank)
    {
	case 0:

            H.sumIntoSystemMatrix(1, 2, val1, ind1);
            H.sumIntoSystemMatrix(2, 2, val2, ind1);

            H.sumIntoSystemMatrix(2, 2, val1, ind2);
            H.sumIntoSystemMatrix(3, 2, val2, ind2);

            H.sumIntoSystemMatrix(3, 2, val1, ind3);

            H.sumIntoRHSVector(2, ind1, val1);   // rhs vector
            H.sumIntoRHSVector(2, ind2, val1);

            H.putIntoSolnVector(3, indg1, valg); // initial guess

	    break;

	case 1:

            H.sumIntoSystemMatrix(4, 2, val2, ind3);

            H.sumIntoSystemMatrix(4, 2, val1, ind4);
            H.sumIntoSystemMatrix(5, 2, val2, ind4);

            H.sumIntoSystemMatrix(5, 2, val1, ind5);
            H.sumIntoSystemMatrix(6, 2, val2, ind5);

            //H.sumIntoRHSVector(2, ind5, val1);   // rhs vector
            //H.putIntoSolnVector(3, indg2, valg); // initial guess//BUG

	    break;

        default:
	    assert(0);
    }

    H.matrixLoadComplete();

    H.selectSolver("pcg");
    H.selectPreconditioner("diagonal");

    H.launchSolver(&status);
    assert(status == 0);

    // get the result
    for (i=1; i<=3; i++)
      if (my_rank == 0)
	printf("sol(%d): %f\n", i, H.accessSolnVector(i));
      else
	printf("sol(%d): %f\n", i+3, H.accessSolnVector(i+3));

    H.resetMatrixAndVector(0.0);
    
    MPI_Finalize();

    // note implicit call to destructor at end of scope
}
