#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <fstream.h>
#include <math.h>


#if 0 // was ifdef SER
#include "mpiuni.h" // was mpiuni/mpi.h
#else
#include "mpi.h"
#endif

// need something to include all the isis mv package
// serial version

#include "other/basicTypes.h"
#include "RealArray.h"
#include "IntArray.h"
#include "GlobalIDArray.h"
#include "CommInfo.h"
#include "Map.h"
#include "Vector.h"
#include "Matrix.h"


// include the Hypre package header here



// to get these isis and fei includes, search from isis root and fei root
// we'll search from fei root and isis-includes
// need these objects to link to later

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
    BASE_SLE(PASSED_COMM_WORLD, masterRank) {

}

//------------------------------------------------------------------------------
HYPRE_SLE::~HYPRE_SLE() {
//
//  Destructor function. Free allocated memory, etc.
//

    // needed here
    deleteLinearAlgebraCore();

}


//------------------------------------------------------------------------------
void HYPRE_SLE::selectSolver(char *name)
{
   
}

//------------------------------------------------------------------------------
void HYPRE_SLE::selectPreconditioner(char *name)
{

    if (!strcmp(name, "identity")) 
    {
    }
    else if (!strcmp(name, "diagonal")) 
    {
    }
    else
    {
    }
}

//------------------------------------------------------------------------------
void HYPRE_SLE::initLinearAlgebraCore(){
//
// This function is called by the constructor, just initializes
// the pointers and other variables associated with the linear
// algebra core to NULL or appropriate initial values.

}

//------------------------------------------------------------------------------
void HYPRE_SLE::deleteLinearAlgebraCore(){
//
//This function deletes allocated memory associated with
//the linear algebra core objects/data structures.
//This is a destructor-type function.
//

}

//------------------------------------------------------------------------------
void HYPRE_SLE::createLinearAlgebraCore(int globalNumEqns,
                                       int localStartRow,
                                       int localEndRow,
                                       int localStartCol,
                                       int localEndCol){
//
//This function is where we establish the structures/objects associated
//with the linear algebra library. i.e., do initial allocations, etc.
//

}

//------------------------------------------------------------------------------
void HYPRE_SLE::matrixConfigure(IntArray* sysRowLengths){

    //first, store the system row-lengths in an ISIS++ Dist_IntVector,
    //to be used in the 'configure' function on the matrix A_.
#if 0
    for (int i = 0; i < storeNumProcEqns; i++) {
        (*rowLengths_)[localStartRow_ + i] = sysRowLengths[i].size();
    }
#endif

    // so now we know all the row lengths, and have them in an IntVector,
    // so we can configure our matrix (not necessary if it is a resizable
    // matrix).

#if 0
    A_ptr_->configure(*rowLengths_);
#endif
}

//------------------------------------------------------------------------------
void HYPRE_SLE::resetMatrixAndVector(double s){

}

//------------------------------------------------------------------------------
void HYPRE_SLE::sumIntoRHSVector(int num, const int* indices, 
                                const double* values){
//
//This function scatters (accumulates) values into the linear-system's
//currently selected RHS vector.
//
// num is how many values are being passed,
// indices holds the global 'row-numbers' into which the values go,
// and values holds the actual coefficients to be scattered.
//

#if 0
    if ((numRHSs_ == 0) && (b_ptr_ == NULL))return;

    for(int i=0; i<num; i++){
        (*b_ptr_)[indices[i]] += values[i];
    }
#endif
}

//------------------------------------------------------------------------------
void HYPRE_SLE::putIntoSolnVector(int num, const int* indices,
                                 const double* values){
//
//This function scatters (puts) values into the linear-system's soln vector.
//
// num is how many values are being passed,
// indices holds the global 'row-numbers' into which the values go,
// and values holds the actual coefficients to be scattered.
//

#if 0
    for(int i=0; i<num; i++){
        (*x_)[indices[i]] = values[i];
    }
#endif
}

//------------------------------------------------------------------------------
double HYPRE_SLE::accessSolnVector(int equation){
//
//This function provides access to the solution vector -- the
//return value is the coefficient at equation 'equation'.
//
// 'equation' must be a 1-based global equation-number.
//
#if 0
    return((*x_)[equation]);
#endif
    return 0.0;
}

//------------------------------------------------------------------------------
void HYPRE_SLE::sumIntoSystemMatrix(int row, int numValues, 
                                   const double* values,
                                   const int* scatterIndices){
//
//This function scatters a row of an element-stiffness array into
//the global system matrix.
//
//row is a global 1-based row-number.
//numValues is how many 'non-zeros' are being passed in.
//values holds the coefficients,
//scatterIndices holds the 1-based global column-indices.
//
#if 0
    A_ptr_->sumIntoRow(row, numValues, values, scatterIndices);
#endif
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
void HYPRE_SLE::matrixLoadComplete(){
//
//This function simply tells the matrix that it's done being
//loaded. Now the matrix can do internal calculations related to
//inter-processor communications, etc.
//
}

//------------------------------------------------------------------------------
void HYPRE_SLE::launchSolver(int* solveStatus){
//
//This function does any last-second setup required for the
//linear solver, then goes ahead and launches the solver to get
//the solution vector.
//Also, if possible, the number of iterations that were performed
//is stored in the iterations_ variable.
//

#if 0
    LinearEquations lse(*A_ptr_, *x_, *b_ptr_);

    if (colScale_) lse.colScale();
    if (rowScale_) lse.rowScale();

    lse.setSolver(*pSolver_);
    lse.setPreconditioner(*pPrecond_);

    *solveStatus = lse.solve();

    iterations_ = pSolver_->iterations();
#endif

}

