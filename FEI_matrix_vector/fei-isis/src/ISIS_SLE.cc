#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef FEI_SER
#include <mpiuni/mpi.h>
#include <isis-ser.h>
#else
#include <mpi.h>
#include <isis-mpi.h>
#endif


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
#include "src/ISIS_SLE.h"

//CASC#include "pc/SAILS_PC.h"
#ifdef HYPRE
#include "pc/PILUT_PC.h"
#endif

//------------------------------------------------------------------------------
ISIS_SLE::ISIS_SLE(MPI_Comm PASSED_COMM_WORLD, int masterRank) : 
    BASE_SLE(PASSED_COMM_WORLD, masterRank) {

//  start the wall clock time recording

    baseTime_ = MPI_Wtime();
    wTime_ = 0.0;
    sTime_ = 0.0;

// currently, not much happens in the constructor....

    pSolver_ = NULL;
    solverAllocated_ = false;

    pPrecond_ = NULL;
    precondAllocated_ = false;

    internalFei_ = 0;

    //default to one rhs vector.
    numRHSs_ = 1;
    rhsIDs_ = new int[numRHSs_];
    rhsIDs_[0] = 0;
    currentRHS_ = 0;

//  and the time spent in the constructor is...

    wTime_  = MPI_Wtime() - baseTime_;

    return;
}

//------------------------------------------------------------------------------
ISIS_SLE::~ISIS_SLE() {
//
//  Destructor function. Free allocated memory, etc.
//

    delete [] rhsIDs_;

    //delete the linear algebra stuff -- matrix pointers, etc.
    deleteLinearAlgebraCore();

    return;
}

//------------------------------------------------------------------------------
void ISIS_SLE::parameters(int numParams, char **paramStrings) {
//
// this function takes parameters for setting internal things like solver
// and preconditioner choice, etc.
//
    baseTime_ = MPI_Wtime();

    if (debugOutput_) {
        fprintf(debugFile_,"ISIS_SLE: parameters\n");
        fflush(debugFile_);
    }

    if (numParams == 0 || paramStrings == NULL) {

        if (debugOutput_) {
            fprintf(debugFile_, "--- no parameters.\n");
        }
    }
    else {
        // take a copy of these parameters, for possible later use.
        appendParamStrings(numParams, paramStrings);

        char param[256];

        if ( getParam("outputLevel",numParams,paramStrings,param) == 1){
            sscanf(param,"%d", &outputLevel_);
        }

        if ( getParam("internalFei",numParams,paramStrings,param) == 1){
            sscanf(param,"%d", &internalFei_);
        }

        if ( getParam("debugOutput",numParams,paramStrings,param) == 1){
            char *name = new char[32];
            sprintf(name, "ISIS_SLE%d_debug", internalFei_);
            setDebugOutput(param,name);
            delete [] name;
        }

        if (debugOutput_) {
           fprintf(debugFile_,"--- numParams %d\n",numParams);
           for(int i=0; i<numParams; i++){
               fprintf(debugFile_,"------ paramStrings[%d]: %s\n",i,
                       paramStrings[i]);
           }
        }
    }

    BASE_SLE::parameters(numParams, paramStrings);

    if (debugOutput_) {
        fprintf(debugFile_,"leaving parameters function\n");
        fflush(debugFile_);
    }

    wTime_ += MPI_Wtime() - baseTime_;

    return;
}

//------------------------------------------------------------------------------
void ISIS_SLE::getMatrixPtr(Matrix** mat){
    *mat = A_ptr_;
}

//------------------------------------------------------------------------------
void ISIS_SLE::getISISMatrixPtr(DCRS_Matrix** mat){
    *mat = A_ptr_;
}

//------------------------------------------------------------------------------
void ISIS_SLE::getRHSVectorPtr(Vector** vec){

    *vec = b_ptr_;
}

//------------------------------------------------------------------------------
void ISIS_SLE::getSolnVectorPtr(Vector** vec){
    *vec = x_;
}

//------------------------------------------------------------------------------
void ISIS_SLE::setNumRHSVectors(int numRHSs, int* rhsIDs){

    if (numRHSs < 0) {
        cerr << "ISIS_SLE::setNumRHSVectors: ERROR, numRHSs < 0." << endl;
    }

    if (numRHSs <= 0) {
        delete [] rhsIDs_;
        rhsIDs_ = NULL;
        numRHSs_ = 0;
        return;
    }

    delete [] rhsIDs_;
    numRHSs_ = numRHSs;
    rhsIDs_ = new int[numRHSs_];

    for(int i=0; i<numRHSs_; i++) rhsIDs_[i] = rhsIDs[i];
}

//------------------------------------------------------------------------------
int ISIS_SLE::setRHSID(int rhsID){

    for(int i=0; i<numRHSs_; i++){
        if (rhsIDs_[i] == rhsID){
            currentRHS_ = i;
            setRHSPtr(b_[currentRHS_]);
            return(0);
        }
    }

    cout << "ISIS_SLE::setRHSID: ERROR, ID not found."
         << endl << "Aborting." << endl;
    abort();

    return(1);
}

//------------------------------------------------------------------------------
void ISIS_SLE::setRHSIndex(int index) {

    if ((index < 0) || (index >= numRHSs_)) {
        cerr << "ISIS_SLE::setRHSIndex: ERROR, index out of range." << endl;
        abort();
    }

    currentRHS_ = index;
    setRHSPtr(b_[currentRHS_]);
}

//------------------------------------------------------------------------------
void ISIS_SLE::setMatrixPtr(DCRS_Matrix* mat) {
    A_ptr_ = mat;
}

//------------------------------------------------------------------------------
void ISIS_SLE::setRHSPtr(Vector* vec) {
    b_ptr_ = vec;
}

//------------------------------------------------------------------------------
void ISIS_SLE::selectSolver(char *name){
   
    if (solverAllocated_) delete pSolver_;

    if (!strcmp(name, "qmr")) {
        pSolver_ = new QMR_Solver;
        if (debugOutput_) {
            fprintf(debugFile_,"QMR solver selected.\n");
        }
    }
    else if (!strcmp(name, "gmres")) {
        pSolver_ = new GMRES_Solver();
        if (debugOutput_) {
            fprintf(debugFile_,"GMRES solver selected.\n");
        }
    }
    else if (!strcmp(name, "fgmres")) {
        pSolver_ = new FGMRES_Solver();
        if (debugOutput_) {
            fprintf(debugFile_,"FGMRES solver selected.\n");
        }
    }
    else if (!strcmp(name, "cg")) {
        pSolver_ = new CG_Solver;
        if (debugOutput_) {
            fprintf(debugFile_,"CG solver selected.\n");
        }
    }
    else if (!strcmp(name, "defgmres")) {
        pSolver_ = new DefGMRES_Solver();
        if (debugOutput_) {
            fprintf(debugFile_,"DefGMRES solver selected.");
        }
    }
    else if (!strcmp(name, "bicgstab")) {
        pSolver_ = new BiCGStab_Solver;
        if (debugOutput_) {
            fprintf(debugFile_,"BiCGStab solver selected.\n");
        }
    }
    else if (!strcmp(name, "cgs")) {
        pSolver_ = new CGS_Solver;
        if (debugOutput_) {
            fprintf(debugFile_,"CGS solver selected.\n");
        }
    }
    else {
        pSolver_ = new QMR_Solver;
        if (debugOutput_) {
            fprintf(debugFile_,"Defaulting to QMR solver.\n");
        }
    }

    solverAllocated_ = true;

    return;
}

//------------------------------------------------------------------------------
void ISIS_SLE::selectPreconditioner(char *name){

    if (precondAllocated_) delete pPrecond_;

    if (!strcmp(name, "identity")) {
        pPrecond_ = new Identity_PC(*A_ptr_);
        if (debugOutput_) {
            fprintf(debugFile_,"Identity pc selected.\n");
        }
    }
    else if (!strcmp(name, "diagonal")) {
        pPrecond_ = new Diagonal_PC(*A_ptr_, *x_);
        if (debugOutput_) {
            fprintf(debugFile_,"Diagonal pc selected.\n");
        }
    }
    else if (!strcmp(name, "polynomial")) {
        pPrecond_ = new Poly_PC(*A_ptr_, *x_);
        if (debugOutput_) {
            fprintf(debugFile_,"Polynomial pc selected.\n");
        }
    }
    else if (!strcmp(name, "bj")) {
        pPrecond_ = new BlockJacobi_PC(*A_ptr_);
        if (debugOutput_) {
            fprintf(debugFile_,"BlockJacobi pc selected.\n");
        }
    }
    else if (!strcmp(name, "spai")) {
        pPrecond_ = new SPAI_PC(*A_ptr_);
        if (debugOutput_) {
            fprintf(debugFile_,"SPAI pc selected.\n");
        }
    }

//CASC   else if (!strcmp(name, "SAILS")) {
//CASC        pPrecond_ = new SAILS_PC(*A_ptr_);
//CASC   if (debugOutput_) {
//CASC        fprintf(debugFile_,"SAILS pc selected.\n");
//CASC   }
//CASC   }
#ifdef HYPRE
    else if (!strcmp(name, "PILUT")) {
        pPrecond_ = new PILUT_PC(*A_ptr_);
       if (debugOutput_) {
           fprintf(debugFile_,"PILUT pc selected.\n");
       }
   }
#endif

    else {
        pPrecond_ = new Identity_PC(*A_ptr_);
        if (debugOutput_) {
            fprintf(debugFile_,"Defaulting to Identity pc.\n");
        }
    }
   
    precondAllocated_ = true;

    return;
}

//------------------------------------------------------------------------------
void ISIS_SLE::initLinearAlgebraCore(){
//
// This function is called by the constructor, just initializes
// the pointers and other variables associated with the linear
// algebra core to NULL or appropriate initial values.
//

    A_ = NULL;
    A_ptr_ = NULL;
    x_ = NULL;
    b_ = NULL;
    b_ptr_ = NULL;
    rowLengths_ = NULL;

    map_ = NULL;
    commInfo_ = NULL;

//  set some defaults

    rowScale_ = false;
    colScale_ = false;

    pSolver_ = NULL;
    pPrecond_ = NULL;

    iterations_ = 0;

}

//------------------------------------------------------------------------------
void ISIS_SLE::deleteLinearAlgebraCore(){
//
//This function deletes allocated memory associated with
//the linear algebra core objects/data structures.
//This is a destructor-type function.
//

    delete A_;
    delete x_;

    for(int i=0; i<numRHSs_; i++) delete b_[i];
    delete [] b_;

    delete map_;
    delete rowLengths_;
    delete commInfo_;

    delete pSolver_;
    delete pPrecond_;
}

//------------------------------------------------------------------------------
void ISIS_SLE::createLinearAlgebraCore(int globalNumEqns,
                                       int localStartRow,
                                       int localEndRow,
                                       int localStartCol,
                                       int localEndCol){
//
//This function is where we establish the structures/objects associated
//with the linear algebra library. i.e., do initial allocations, etc.
//
    (void)localStartCol;
    (void)localEndCol;

    if (debugOutput_) {
        fprintf(debugFile_,"createLinearAlgebraCore: numRHSs_: %d\n",numRHSs_);
        fflush(debugFile_);
    }

//  construct a CommInfo object.
    
    commInfo_ = new CommInfo(masterRank_, FEI_COMM_WORLD);

    map_ = new Map(globalNumEqns, localStartRow, localEndRow, *commInfo_);
    A_ = new DCRS_Matrix(*map_);
    x_ = new Dist_Vector(*map_);
    if (numRHSs_ > 0)
        b_ = new Dist_Vector*[numRHSs_];
    for(int i=0; i<numRHSs_; i++){
        b_[i] = new Dist_Vector(*map_);
        b_[i]->put(0.0);
    }

    if (A_ptr_ == NULL) A_ptr_ = A_;
    if ((b_ptr_ == NULL) && (numRHSs_ > 0)) b_ptr_ = b_[currentRHS_];

    rowLengths_ = new Dist_IntVector(*map_);
}

//------------------------------------------------------------------------------
void ISIS_SLE::matrixConfigure(IntArray* sysRowLengths){

    //first, store the system row-lengths in an ISIS++ Dist_IntVector,
    //to be used in the 'configure' function on the matrix A_.
    for (int i = 0; i < storeNumProcEqns; i++) {
        (*rowLengths_)[localStartRow_ + i] = sysRowLengths[i].size();
    }

    // so now we know all the row lengths, and have them in an IntVector,
    // so we can configure our matrix (not necessary if it is a resizable
    // matrix).

    A_ptr_->configure(*rowLengths_);
    
}

//------------------------------------------------------------------------------
void ISIS_SLE::resetMatrixAndVector(double s){

    A_ptr_->put(s);
    if (A_ptr_ != A_) A_->put(s);

    for(int i=0; i<numRHSs_; i++){
        b_[i]->put(s);
    }
    b_ptr_->put(s);
}

//------------------------------------------------------------------------------
void ISIS_SLE::sumIntoRHSVector(int num, const int* indices, 
                                const double* values){
//
//This function scatters (accumulates) values into the linear-system's
//currently selected RHS vector.
//
// num is how many values are being passed,
// indices holds the global 'row-numbers' into which the values go,
// and values holds the actual coefficients to be scattered.
//
    if (debugOutput_) {
        fprintf(debugFile_,"sumIntoRHSVector: currentRHS_: %d\n",
                currentRHS_);
        fflush(debugFile_);
    }

    if ((numRHSs_ == 0) && (b_ptr_ == NULL))return;

    for(int i=0; i<num; i++){
        (*b_ptr_)[indices[i]] += values[i];
    }
}

//------------------------------------------------------------------------------
void ISIS_SLE::putIntoSolnVector(int num, const int* indices,
                                 const double* values){
//
//This function scatters (puts) values into the linear-system's soln vector.
//
// num is how many values are being passed,
// indices holds the global 'row-numbers' into which the values go,
// and values holds the actual coefficients to be scattered.
//

    for(int i=0; i<num; i++){
        (*x_)[indices[i]] = values[i];
    }
}

//------------------------------------------------------------------------------
double ISIS_SLE::accessSolnVector(int equation){
//
//This function provides access to the solution vector -- the
//return value is the coefficient at equation 'equation'.
//
// 'equation' must be a 1-based global equation-number.
//
    return((*x_)[equation]);
}

//------------------------------------------------------------------------------
void ISIS_SLE::sumIntoSystemMatrix(int row, int numValues, 
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
    A_ptr_->sumIntoRow(row, numValues, values, scatterIndices);
}

//------------------------------------------------------------------------------
void ISIS_SLE::enforceEssentialBC(int* globalEqn, double* alpha,
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

    double* coefs = NULL;
    int* indices = NULL;
    int rowLength = 0;

    for(int i=0; i<len; i++) {

       if ((localStartRow_ <= globalEqn[i]) && (globalEqn[i] <= localEndRow_)){
        int localEqn = globalEqn[i] - localStartRow_;

        for(int jj=0; jj<sysMatIndices[localEqn].size(); jj++) {

            int row = (sysMatIndices[localEqn])[jj];

            if ((localStartRow_ <= row) && (row <= localEndRow_)) {
                rowLength = A_ptr_->rowLength(row);
                coefs = A_ptr_->getPointerToCoef(rowLength, row);
                indices = A_ptr_->getPointerToColIndex(rowLength, row);

                if (row==globalEqn[i]) {
                    //if this is row globalEqn[i], zero it and leave a 1.0 on
                    //the diagonal.

                    int j;
                    for(j=0; j<rowLength; j++) {
                        if (indices[j] == row) coefs[j] = 1.0;
                        else coefs[j] = 0.0;
                    }

                    double rhs_term = gamma[i]/alpha[i];

//                    //also, make the rhs modification here.
//                    for(j=0; j<numRHSs_; j++) {
//                        (*(b_[j]))[row] = rhs_term;
//                    }
//                    if (b_ptr_ != b_[currentRHS_])
                        (*b_ptr_)[row] = rhs_term;
                }
                else {
                    //else look through the row to find the non-zero in position
                    //globalEqn[i] and make the appropriate modification.

                    int j;
                    for(j=0; j<rowLength; j++) {

                        if (indices[j] == globalEqn[i]) {
                            double rhs_term = gamma[i]/alpha[i];

//                            //make the rhs modification here.
//                            for(int r=0; r<numRHSs_; r++) {
//                                (*(b_[r]))[row] -= coefs[j]*rhs_term;
//                            }
//                            if (b_ptr_ != b_[currentRHS_])
                                (*b_ptr_)[row] -= coefs[j]*rhs_term;

                            coefs[j] = 0.0;

                            break;
                        }
                    }
                }
            }
        }
       }
    }

    return;
}

//------------------------------------------------------------------------------
void ISIS_SLE::enforceOtherBC(int* globalEqn, double* alpha, double* beta,
                              double* gamma, int len) {
//
//This function must enforce a natural or mixed boundary condition
//on equation 'globalEqn'. This means that the following modification should
//be made to A and b:
//
//A[globalEqn,globalEqn] += alpha/beta;
//b[globalEqn] += gamma/beta;
//

    for(int i=0; i<len; i++) {
        if ((globalEqn[i] < localStartRow_) || (globalEqn[i] > localEndRow_))
            break;

        int rowLength = 0;
        double* coefs = A_ptr_->getPointerToCoef(rowLength, globalEqn[i]);
        int* indices = A_ptr_->getPointerToColIndex(rowLength, globalEqn[i]);

        for(int j=0; j<rowLength; j++) {
            if (indices[j] == globalEqn[i]) {
                coefs[j] += alpha[i]/beta[i];

                break;
            }
        }

        //now make the rhs modification.
//        (*(b_[currentRHS_]))[globalEqn[i]] += gamma[i]/beta[i];
//        if (b_ptr_ != b_[currentRHS_])
            (*b_ptr_)[globalEqn[i]] += gamma[i]/beta[i];
    }

    return;
}

//------------------------------------------------------------------------------
void ISIS_SLE::matrixLoadComplete(){
//
//This function simply tells the matrix that it's done being
//loaded. Now the matrix can do internal calculations related to
//inter-processor communications, etc.
//
    if (debugOutput_) {
        fprintf(debugFile_,"matrixLoadComplete\n");
        fflush(debugFile_);
    }

    A_ptr_->fillComplete();

    if (debugOutput_) {
        fprintf(debugFile_,"leaving matrixLoadComplete\n");
        fflush(debugFile_);
    }
}

//------------------------------------------------------------------------------
void ISIS_SLE::launchSolver(int* solveStatus){
//
//This function does any last-second setup required for the
//linear solver, then goes ahead and launches the solver to get
//the solution vector.
//Also, if possible, the number of iterations that were performed
//is stored in the iterations_ variable.
//

    if (debugOutput_) {
        char matname[256];
        sprintf(matname,"A_ISIS_SLE.mtx.slv%d.np%d", solveCounter_, numProcs_);
        A_ptr_->writeToFile(matname);
        sprintf(matname,"x_ISIS_SLE.txt.slv%d.np%d", solveCounter_, numProcs_);
        x_->writeToFile(matname);
        if (numRHSs_>0) {
            sprintf(matname,"b_ISIS_SLE.txt.slv%d.np%d", solveCounter_,
                    numProcs_);
            b_ptr_->writeToFile(matname);
        }
    }

    char param[64];

    if ( getParam("rowScale",numParams_,paramStrings_,param) == 1){
        if (!strcmp(param,"true")) rowScale_ = true;
    }

    if ( getParam("colScale",numParams_,paramStrings_,param) == 1){
        if (!strcmp(param,"true")) colScale_ = true;
    }

    pSolver_->outputLevel(outputLevel_,localRank_,masterRank_);
    pSolver_->parameters(numParams_,paramStrings_);
    pPrecond_->parameters(numParams_,paramStrings_);

    pPrecond_->calculate();

    LinearEquations lse(*A_ptr_, *x_, *b_ptr_);

    if (colScale_) lse.colScale();
    if (rowScale_) lse.rowScale();

    lse.setSolver(*pSolver_);
    lse.setPreconditioner(*pPrecond_);

    *solveStatus = lse.solve();

    iterations_ = pSolver_->iterations();
}

