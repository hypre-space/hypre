#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>

#include "src/Data.h"

#ifdef FEI_SER
#include <isis-ser.h>
#else
#include <isis-mpi.h>
#endif

#include "src/LinearSystemCore.h"
#include "other/basicTypes.h"
#include "src/Utils.h"

#include "src/ISIS_LinSysCore.h"

//=========CONSTRUCTOR==========================================================
ISIS_LinSysCore::ISIS_LinSysCore(MPI_Comm comm)
 : LinearSystemCore(comm),
   comm_(comm),
   commInfo_(NULL),
   map_(NULL),
   A_(NULL),
   A_ptr_(NULL),
   x_(NULL),
   b_(NULL),
   b_ptr_(NULL),
   matricesVectorsCreated_(false),
   rhsIDs_(NULL),
   numRHSs_(0),
   currentRHS_(-1),
   localStartRow_(0),
   numLocalRows_(0),
   localEndRow_(0),
   pSolver_(NULL),
   solverName_(NULL),
   solverAllocated_(false),
   pPrecond_(NULL),
   precondName_(NULL),
   precondAllocated_(false),
   rowScale_(false),
   colScale_(false),
   solveCounter_(0),
   outputLevel_(0),
   numParams_(0),
   paramStrings_(NULL),
   debugOutput_(0),
   debugFileCounter_(0),
   debugPath_(NULL),
   debugFileName_(NULL),
   debugFile_(NULL),
   dumpMatrix_(false),
   matrixPath_(NULL)
{
   masterProc_ = 0;
   MPI_Comm_size(comm_, &numProcs_);
   MPI_Comm_rank(comm_, &thisProc_);

   numRHSs_ = 1;
   rhsIDs_ = new int[numRHSs_];
   rhsIDs_[0] = 0;

   solverName_ = new char[128];
   sprintf(solverName_,"gmres");
   precondName_ = new char[128];
   sprintf(precondName_,"identity");
}

//========DESTRUCTOR============================================================
ISIS_LinSysCore::~ISIS_LinSysCore() {

   if (matricesVectorsCreated_) {
      delete A_;
      delete x_;
      for(int i=0; i<numRHSs_; i++) {
         delete b_[i];
      }
      delete [] b_;
      delete map_;
      delete commInfo_;
   }

   delete [] rhsIDs_;
   numRHSs_ = 0;

   if (solverAllocated_) {
      delete pSolver_;
      solverAllocated_ = false;
   }

   if (precondAllocated_) {
      delete pPrecond_;
      precondAllocated_ = false;
   }

   for(int i=0; i<numParams_; i++) {
      delete [] paramStrings_[i];
   }
   delete [] paramStrings_;
   numParams_ = 0;

   if (debugOutput_) {
      debugOutput_ = 0;
      fclose(debugFile_);
      delete [] debugPath_;
      delete [] debugFileName_;
   }

   if (dumpMatrix_) {
      delete [] matrixPath_;
   }

   delete [] solverName_;
   delete [] precondName_;
}

//==============================================================================
LinearSystemCore* ISIS_LinSysCore::clone() {
   return(new ISIS_LinSysCore(comm_));
}

//==============================================================================
void ISIS_LinSysCore::parameters(int numParams, char** params) {
//
// this function takes parameters for setting internal things like solver
// and preconditioner choice, etc.
//
   debugOutput("parameters");

   if (numParams == 0 || params == NULL) {

      debugOutput("--- no parameters");
   }
   else {
      char param[256];

      Utils::appendToCharArrayList(paramStrings_, numParams_,
                                   params, numParams);

      if ( Utils::getParam("outputLevel",numParams,params,param) == 1){
         sscanf(param,"%d", &outputLevel_);
      }

      if ( Utils::getParam("dumpMatrix",numParams,params,param) == 1){
         dumpMatrix_ = true;
         delete [] matrixPath_;
         matrixPath_ = new char[strlen(param)+1];
         sprintf(matrixPath_, param);
      }

      if ( Utils::getParam("debugOutput",numParams,params,param) == 1){
         char *name = new char[64];

         sprintf(name, "ISIS_LSC_debug.%d.%d.file%d",
                 numProcs_, thisProc_, debugFileCounter_);
         debugFileCounter_++;

         setDebugOutput(param,name);

         dumpMatrix_ = true;
         delete [] matrixPath_;
         matrixPath_ = new char[strlen(param)+1];
         sprintf(matrixPath_, param);

         delete [] name;
      }

      if( Utils::getParam("solver",numParams, params, param) == 1){
         sprintf(solverName_, param);
      }

      if( Utils::getParam("preconditioner",numParams, params, param) == 1){
         sprintf(precondName_, param);
      }

      if ( Utils::getParam("rowScale",numParams, params, param) == 1){
         if (!strcmp(param,"true")) rowScale_ = true;
      }

      if ( Utils::getParam("colScale",numParams, params, param) == 1){
         if (!strcmp(param,"true")) colScale_ = true;
      }

      if (debugOutput_) {
         fprintf(debugFile_,"--- numParams %d\n",numParams);
         for(int i=0; i<numParams; i++){
            fprintf(debugFile_,"------ paramStrings[%d]: %s\n",i,
                    paramStrings_[i]);
         }
      }
   }

   debugOutput("leaving parameters function");
}

//==============================================================================
void ISIS_LinSysCore::createMatricesAndVectors(int numGlobalEqns,
                                            int firstLocalEqn,
                                            int numLocalEqns) {
//
//This function is where we establish the structures/objects associated
//with the linear algebra library. i.e., do initial allocations, etc.
//
   if (debugOutput_) {
      fprintf(debugFile_,
              "createMatricesVectors: numRHSs_: %d, firstLocalEqn: %d\n",
              numRHSs_, firstLocalEqn);
      fprintf(debugFile_,
              "createMatricesVectors: numLocalEqns: %d, numGlobalEqns: %d\n",
              numLocalEqns, numGlobalEqns);
      fflush(debugFile_);
   }

// construct a CommInfo object.

   commInfo_ = new CommInfo(masterProc_, comm_);

   if (numLocalEqns > numGlobalEqns)
      messageAbort("createMatricesVectors: numLocalEqns > numGlobalEqns");

   if ((0 >= firstLocalEqn) || (firstLocalEqn > numGlobalEqns))
      messageAbort("createMatricesVectors: firstLocalEqn out of range");

   localStartRow_ = firstLocalEqn;
   localEndRow_ = firstLocalEqn + numLocalEqns - 1;
   numLocalRows_ = localEndRow_ - localStartRow_ + 1;

   if (localEndRow_ > numGlobalEqns)
      messageAbort("createMatricesVectors: inconsistent sizes and eqn number.");

   map_ = new Map(numGlobalEqns, localStartRow_, localEndRow_, *commInfo_);

   A_ = new DCRS_Matrix(*map_);
   x_ = new Dist_Vector(*map_);

   if (numRHSs_ <= 0)
      messageAbort("numRHSs_==0. Out of scope or destructor already called?");

   b_ = new Dist_Vector*[numRHSs_];

   for(int i=0; i<numRHSs_; i++){
      b_[i] = new Dist_Vector(*map_);
      b_[i]->put(0.0);
   }

   if (A_ptr_ == NULL) A_ptr_ = A_;

   if (currentRHS_ < 0) currentRHS_ = 0;

   if ((b_ptr_ == NULL) && (numRHSs_ > 0)) b_ptr_ = b_[currentRHS_];

   matricesVectorsCreated_ = true;
}

//==============================================================================
void ISIS_LinSysCore::allocateMatrix(int** colIndices, int* rowLengths) {

   (void)colIndices;

   //first, store the row-lengths in an ISIS++ Dist_IntVector,
   //to be used in the 'configure' function on the matrix A_.

   Dist_IntVector row_lengths(*map_);

   for (int i = 0; i < numLocalRows_; i++) {
      row_lengths[localStartRow_ + i] = rowLengths[i];
   }

   // so now we know all the row lengths, and have them in an IntVector,
   // so we can configure our matrix (not necessary if it is a resizable
   // matrix).

   A_ptr_->configure(row_lengths);
}

//==============================================================================
void ISIS_LinSysCore::resetMatrixAndVector(double s) {

   if (A_ptr_ != NULL) A_ptr_->put(s);

   if (A_ptr_ != A_) A_->put(s);

   if (b_ != NULL) {
      for(int i=0; i<numRHSs_; i++){
         b_[i]->put(s);
      }
   }

   if (b_ptr_ != NULL) b_ptr_->put(s);
}

//==============================================================================
void ISIS_LinSysCore::sumIntoSystemMatrix(int row, int numValues,
                                          const double* values,
                                          const int* scatterIndices) {

   if (A_ptr_ == NULL)
      messageAbort("sumIntoSystemMatrix: matrix is NULL.");

   A_ptr_->sumIntoRow(row, numValues, values, scatterIndices);
}

//==============================================================================
void ISIS_LinSysCore::sumIntoRHSVector(int num, const double* values,
                                       const int* indices) {
//
//This function scatters (accumulates) values into the linear-system's
//currently selected RHS vector.
//
// num is how many values are being passed,
// indices holds the global 'row-numbers' into which the values go,
// and values holds the actual coefficients to be scattered.
//

    if ((numRHSs_ == 0) && (b_ptr_ == NULL))return;

    for(int i=0; i<num; i++){
        (*b_ptr_)[indices[i]] += values[i];
    }
}

//==============================================================================
void ISIS_LinSysCore::matrixLoadComplete() {

    debugOutput("matrixLoadComplete");

    A_ptr_->writeToFile("A_preload.mtx");
    A_ptr_->fillComplete();

    debugOutput("leaving matrixLoadComplete");
}

//==============================================================================
void ISIS_LinSysCore::enforceEssentialBC(int* globalEqn,
                                         double* alpha,
                                         double* gamma, int len) {
//
//This function must enforce an essential boundary condition on each local
//equation in 'globalEqn'. This means, that the following modification
//should be made to A and b, for each globalEqn:
//
//for(each local equation i){
//   for(each column j in row i) {
//      if (i==j) b[i] = gamma/alpha;
//      else b[j] -= (gamma/alpha)*A[j,i];
//   }
//}
//
//all of row 'globalEqn' and column 'globalEqn' in A should be zeroed,
//except for 1.0 on the diagonal.
//

   double* coefs = NULL;
   int* indices = NULL;
   int rowLength = 0;

   for(int i=0; i<len; i++) {

      //if globalEqn[i] is local, we'll diagonalize the row and column.

      if ((localStartRow_ <= globalEqn[i]) && (globalEqn[i] <= localEndRow_)){
         rowLength = A_ptr_->rowLength(globalEqn[i]);
         indices = A_ptr_->getPointerToColIndex(rowLength, globalEqn[i]);
         coefs = A_ptr_->getPointerToCoef(rowLength, globalEqn[i]);

         for(int jj=0; jj<rowLength; jj++) {

            //zero this row, except for the diagonal coefficient.
            if (indices[jj] == globalEqn[i]) coefs[jj] = 1.0;
            else coefs[jj] = 0.0;

            //also, make the rhs modification here.
            double rhs_term = gamma[i]/alpha[i];
            (*b_ptr_)[globalEqn[i]] = rhs_term;

            int col_row = indices[jj];

            if ((localStartRow_ <= col_row) && (col_row <= localEndRow_)) {
               if (col_row != globalEqn[i]) {
                  int thisLen = A_ptr_->rowLength(col_row);
                  int* theseInds = A_ptr_->
                                     getPointerToColIndex(thisLen, col_row);
                  double* theseCoefs = A_ptr_->
                                     getPointerToCoef(thisLen, col_row);

                  //look through the row to find the non-zero in position
                  //globalEqn[i] and make the appropriate modification.

                  for(int j=0; j<thisLen; j++) {

                     if (theseInds[j] == globalEqn[i]) {
                        rhs_term = gamma[i]/alpha[i];

                        (*b_ptr_)[col_row] -= theseCoefs[j]*rhs_term;

                        theseCoefs[j] = 0.0;

                        break;
                     }
                  }
               }
            }
         }// end for(jj<rowLength) loop
      }
   }
}

//==============================================================================
void ISIS_LinSysCore::enforceRemoteEssBCs(int numEqns, int* globalEqns,
                                          int** colIndices, int* colIndLen,
                                          double** coefs) {
//
//globalEqns should hold eqns that are owned locally, but which contain
//column indices (the ones in colIndices) which are from remote equations
//on which essential boundary-conditions need to be enforced.
//
//This function will only make the modification if the above conditions
//hold -- i.e., the equation is a locally-owned equation, and the column
//index is NOT a locally owned equation.
//
   for(int i=0; i<numEqns; i++) {

      if ((globalEqns[i] < localStartRow_) || (globalEqns[i] > localEndRow_)) {
         continue;
      }

      int rowLen = A_ptr_->rowLength(globalEqns[i]);
      int* AcolInds = A_ptr_->getPointerToColIndex(rowLen, globalEqns[i]);
      double* Acoefs = A_ptr_->getPointerToCoef(rowLen, globalEqns[i]);

      for(int j=0; j<colIndLen[i]; j++) {
         for(int k=0; k<rowLen; k++) {
           if ((colIndices[i][j] < localStartRow_) ||
               (colIndices[i][j] > localEndRow_)) {
            if (AcolInds[k] == colIndices[i][j]) {
               (*b_ptr_)[globalEqns[i]] -= Acoefs[k]*coefs[i][j];
               Acoefs[k] = 0.0;
            }
           }
         }
      }
   }
}

//==============================================================================
void ISIS_LinSysCore::enforceOtherBC(int* globalEqn,
                                     double* alpha,
                                     double* beta,
                                     double* gamma,
                                     int len) {
//
//This function must enforce a natural or mixed boundary condition on the
//equations in 'globalEqn'. This means that the following modification should
//be made to A and b:
//
//A[globalEqn,globalEqn] += alpha/beta;
//b[globalEqn] += gamma/beta;
//

   for(int i=0; i<len; i++) {
      if ((globalEqn[i] < localStartRow_) || (globalEqn[i] > localEndRow_)) {
         continue;
      }

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
      (*b_ptr_)[globalEqn[i]] += gamma[i]/beta[i];
   }
}

//==============================================================================
void ISIS_LinSysCore::getMatrixPtr(Data& data) {

   data.setTypeName("DCRS_Matrix");
   data.setDataPtr((void*)A_ptr_);
}

//==============================================================================
void ISIS_LinSysCore::copyInMatrix(double scalar, const Data& data) {
//
//Overwrites the current internal matrix with a scaled copy of the
//input argument.
//
   if (strcmp("DCRS_Matrix", data.getTypeName()))
      messageAbort("copyInMatrix: data's type string not 'DCRS_Matrix'.");

   DCRS_Matrix* source = (DCRS_Matrix*)data.getDataPtr();

   A_ptr_->copyScaledMatrix(scalar, *source);
}

//==============================================================================
void ISIS_LinSysCore::copyOutMatrix(double scalar, Data& data) {
//
//Passes out a scaled copy of the current internal matrix.
//

   DCRS_Matrix* outmat = new DCRS_Matrix(*map_);

   outmat->copyScaledMatrix(scalar, *A_ptr_);

   data.setTypeName("DCRS_Matrix");
   data.setDataPtr((void*)outmat);
}

//==============================================================================
void ISIS_LinSysCore::sumInMatrix(double scalar, const Data& data) {

   if (strcmp("DCRS_Matrix", data.getTypeName()))
      messageAbort("sumInMatrix: data's type string not 'DCRS_Matrix'.");

   DCRS_Matrix* source = (DCRS_Matrix*)data.getDataPtr();

   A_ptr_->addScaledMatrix(scalar, *source);
}

//==============================================================================
void ISIS_LinSysCore::getRHSVectorPtr(Data& data) {

   data.setTypeName("Dist_Vector");
   data.setDataPtr((void*)b_ptr_);
}

//==============================================================================
void ISIS_LinSysCore::copyInRHSVector(double scalar, const Data& data) {

   if (strcmp("Dist_Vector", data.getTypeName()))
      messageAbort("copyInRHSVector: data's type string not 'Dist_Vector'.");

   Dist_Vector* sourcevec = (Dist_Vector*)data.getDataPtr();

   *b_ptr_ = *sourcevec;

   if (scalar != 1.0) b_ptr_->scale(scalar);
}

//==============================================================================
void ISIS_LinSysCore::copyOutRHSVector(double scalar, Data& data) {

   Dist_Vector* outvec = new Dist_Vector(*map_);

   //the Dist_Vector constructor initializes the new vector with 0.0

   outvec->addVec(scalar, *b_ptr_);

   data.setTypeName("Dist_Vector");
   data.setDataPtr((void*)outvec);
}

//==============================================================================
void ISIS_LinSysCore::sumInRHSVector(double scalar, const Data& data) {

   if (strcmp("Dist_Vector", data.getTypeName()))
      messageAbort("sumInRHSVector: data's type string not 'Dist_Vector'.");

   Dist_Vector* source = (Dist_Vector*)data.getDataPtr();

   b_ptr_->addVec(scalar, *source);
}

//==============================================================================
void ISIS_LinSysCore::destroyMatrixData(Data& data) {

   if (strcmp("DCRS_Matrix", data.getTypeName()))
      messageAbort("destroyMatrixData: data doesn't contain a DCRS_Matrix.");

   DCRS_Matrix* mat = (DCRS_Matrix*)data.getDataPtr();
   delete mat;
}

//==============================================================================
void ISIS_LinSysCore::destroyVectorData(Data& data) {

   if (strcmp("Dist_Vector", data.getTypeName()))
      messageAbort("destroyVectorData: data doesn't contain a Dist_Vector.");

   Dist_Vector* vec = (Dist_Vector*)data.getDataPtr();
   delete vec;
}

//==============================================================================
void ISIS_LinSysCore::setNumRHSVectors(int numRHSs, const int* rhsIDs) {

   if (numRHSs < 0)
      messageAbort("setNumRHSVectors: numRHSs < 0.");

   if (numRHSs == 0) return;

   delete [] rhsIDs_;
   numRHSs_ = numRHSs;
   rhsIDs_ = new int[numRHSs_];

   for(int i=0; i<numRHSs_; i++) rhsIDs_[i] = rhsIDs[i];
}

//==============================================================================
void ISIS_LinSysCore::setRHSID(int rhsID) {

    for(int i=0; i<numRHSs_; i++){
        if (rhsIDs_[i] == rhsID){
            currentRHS_ = i;
            b_ptr_ = b_[currentRHS_];
            return;
        }
    }

    messageAbort("setRHSID: rhsID not found.");
}

//==============================================================================
void ISIS_LinSysCore::putInitialGuess(const int* eqnNumbers,
                                      const double* values,
                                      int len) {
//
//This function scatters (puts) values into the linear-system's soln vector.
//
// num is how many values are being passed,
// indices holds the global 'row-numbers' into which the values go,
// and values holds the actual coefficients to be scattered.
//

   for(int i=0; i<len; i++){
      if ((localStartRow_ > eqnNumbers[i]) || (eqnNumbers[i] > localEndRow_)) {
         cout << "localStartRow_: " << localStartRow_ << ", localEndRow_: " 
            << localEndRow_ << ", eqnNumbers["<<i<<"]: " << eqnNumbers[i] 
           << endl;
         messageAbort("putInitialGuess: eqnNumber out of range.");
      }

      (*x_)[eqnNumbers[i]] = values[i];
   }
}

//==============================================================================
void ISIS_LinSysCore::getSolution(int* eqnNumbers, double* answers, int len) {
//
//The caller must allocate the memory for 'answers' and eqnNumbers,
//and len must be set to the right value -- i.e., len should equal
//numLocalRows_.
//
   if (len != numLocalRows_)
      messageAbort("getSolution: len != numLocalRows_.");

   for(int i=0; i<numLocalRows_; i++) {
      answers[i] = (*x_)[localStartRow_ + i];
      eqnNumbers[i] = localStartRow_ + i;
   }
}

//==============================================================================
void ISIS_LinSysCore::getSolnEntry(int eqnNumber, double& answer) {
//
//This function returns a single solution entry, the coefficient for
//equation number eqnNumber.
//
   if ((localStartRow_ > eqnNumber) || (eqnNumber > localEndRow_))
      messageAbort("getSolnEntry: eqnNumber out of range.");

   answer = (*x_)[eqnNumber];
}

//==============================================================================
void ISIS_LinSysCore::selectSolver(char* name) {
  
    if (solverAllocated_) delete pSolver_;

    char outString[64];

    if (!strcmp(name, "qmr")) {
        pSolver_ = new QMR_Solver;
        sprintf(outString,"selectSolver: QMR selected.\n");
    }
    else if (!strcmp(name, "gmres")) {
        pSolver_ = new GMRES_Solver();
        sprintf(outString,"selectSolver: GMRES selected.\n");
    }
    else if (!strcmp(name, "fgmres")) {
        pSolver_ = new FGMRES_Solver();
        sprintf(outString,"selectSolver: FGMRES selected.\n");
    }
    else if (!strcmp(name, "cg")) {
        pSolver_ = new CG_Solver;
        sprintf(outString,"selectSolver: CG selected.\n");
    }
    else if (!strcmp(name, "defgmres")) {
        pSolver_ = new DefGMRES_Solver();
        sprintf(outString,"selectSolver: DefGMRES selected.\n");
    }
    else if (!strcmp(name, "bicgstab")) {
        pSolver_ = new BiCGStab_Solver;
        sprintf(outString,"selectSolver: BiCGStab selected.\n");
    }
    else if (!strcmp(name, "cgs")) {
        pSolver_ = new CGS_Solver;
        sprintf(outString,"selectSolver: CGS selected.\n");
    }
    else {
        pSolver_ = new QMR_Solver;
        sprintf(outString,"selectSolver: Defaulting to QMR.\n");
    }

   debugOutput(outString);

   solverAllocated_ = true;
}

//==============================================================================
void ISIS_LinSysCore::selectPreconditioner(char* name) {

   if (precondAllocated_) delete pPrecond_;

   char outString[64];

   if (!strcmp(name, "identity")) {
      pPrecond_ = new Identity_PC(*A_ptr_);
      sprintf(outString,"selectPreconditioner: Identity selected.\n");
   }
   else if (!strcmp(name, "diagonal")) {
      pPrecond_ = new Diagonal_PC(*A_ptr_, *x_);
      sprintf(outString,"selectPreconditioner: Diagonal selected.\n");
   }
   else if (!strcmp(name, "polynomial")) {
      pPrecond_ = new Poly_PC(*A_ptr_, *x_);
      sprintf(outString,"selectPreconditioner: Polynomial selected.\n");
   }
   else if (!strcmp(name, "bj")) {
      pPrecond_ = new BlockJacobi_PC(*A_ptr_);
      sprintf(outString,"selectPreconditioner: BlockJacobi selected.\n");
   }
   else if (!strcmp(name, "spai")) {
      pPrecond_ = new SPAI_PC(*A_ptr_);
      sprintf(outString,"selectPreconditioner: SPAI selected.\n");
   }
   else {
      pPrecond_ = new Identity_PC(*A_ptr_);
      sprintf(outString,"selectPreconditioner: Defaulting to Identity.\n");
   }

   debugOutput(outString);

   precondAllocated_ = true;
}

//==============================================================================
void ISIS_LinSysCore::writeSystem(char* name) {
   char matname[256];
   sprintf(matname,"%s/A_%s.mtx.slv%d.np%d", debugPath_, name, solveCounter_,
           numProcs_);
   A_ptr_->writeToFile(matname);
   if (numRHSs_>0) {
      sprintf(matname,"%s/b_%s.txt.slv%d.np%d", debugPath_, name,
              solveCounter_, numProcs_);
      b_ptr_->writeToFile(matname);
   }
}

//==============================================================================
void ISIS_LinSysCore::launchSolver(int& solveStatus, int& iterations) {
//
//This function does any last-second setup required for the
//linear solver, then goes ahead and launches the solver to get
//the solution vector.
//Also, if possible, the number of iterations that were performed
//is stored in the iterations_ variable.
//
   debugOutput("launchSolver");

   solveCounter_++;

   if (dumpMatrix_) {
      char matname[256];
      sprintf(matname,"%s/A_ISIS.mtx.slv%d.np%d", matrixPath_, solveCounter_,
              numProcs_);
      A_ptr_->writeToFile(matname);
      sprintf(matname,"%s/x_ISIS.txt.pre-slv%d.np%d", matrixPath_,
              solveCounter_, numProcs_);
      x_->writeToFile(matname);
      if (numRHSs_>0) {
         sprintf(matname,"%s/b_ISIS.txt.slv%d.np%d", matrixPath_, solveCounter_,
                 numProcs_);
         b_ptr_->writeToFile(matname);
      }
   }

   debugOutput("setting solver and precond. parameters");
   selectSolver(solverName_);
   selectPreconditioner(precondName_);

   if (outputLevel_ > 0) {
      cout << "ISIS_LinSysCore: Solver: " << solverName_ << endl;
      cout << "ISIS_LinSysCore: Precond: " << precondName_ << endl;
   }

   pSolver_->outputLevel(outputLevel_, thisProc_, masterProc_);
   pSolver_->parameters(numParams_, paramStrings_);
   pPrecond_->parameters(numParams_, paramStrings_);

   debugOutput("calling pPrecond_->calculate() ...");

   pPrecond_->calculate();

   debugOutput("...back from pPrecond_->calculate()");

   LinearEquations lse(*A_ptr_, *x_, *b_ptr_);

   if (colScale_) lse.colScale();
   if (rowScale_) lse.rowScale();

   lse.setSolver(*pSolver_);
   lse.setPreconditioner(*pPrecond_);

   debugOutput("launching the solver...");
   solveStatus = lse.solve();
   debugOutput("...back from the solver");

   iterations = pSolver_->iterations();

   if (dumpMatrix_) {
      char vecname[256];
      sprintf(vecname,"%s/x_ISIS.txt.soln%d.np%d", matrixPath_, solveCounter_,
              numProcs_);
      x_->writeToFile(vecname);
   }

   debugOutput("leaving launchSolver");
}

//==============================================================================
void ISIS_LinSysCore::setDebugOutput(char* path, char* name){
//
//This function turns on debug output, and opens a file to put it in.
//
   if (debugOutput_) {
      fprintf(debugFile_,"setDebugOutput closing this file.");
      fflush(debugFile_);
      fclose(debugFile_);
      debugFile_ = NULL;
   }

   int pathLength = strlen(path);
   if (path != debugPath_) {
      delete [] debugPath_;
      debugPath_ = new char[pathLength + 1];
      sprintf(debugPath_, path);
   }

   int nameLength = strlen(name);
   if (name != debugFileName_) {
      delete [] debugFileName_;
      debugFileName_ = new char[nameLength + 1];
      sprintf(debugFileName_,name);
   }

   char* dbFileName = new char[pathLength + nameLength + 3];

   sprintf(dbFileName, "%s/%s", path, name);

   debugOutput_ = 1;
   debugFile_ = fopen(dbFileName, "w");

   if (!debugFile_){
      cerr << "couldn't open debug output file: " << debugFileName_ << endl;
      debugOutput_ = 0;
   }

   delete [] dbFileName;
}

//==============================================================================
void ISIS_LinSysCore::debugOutput(char* mesg) const {
   if (debugOutput_) {
      fprintf(debugFile_, "%s\n",mesg);
      fflush(debugFile_);
   }
}

//==============================================================================
void ISIS_LinSysCore::messageAbort(char* msg) const {
   cerr << "ISIS_LinSysCore: " << msg << " Aborting." << endl;
   abort();
}

