#ifndef __ISIS_SLE_H
#define __ISIS_SLE_H

// #include "fei.h"
// #include "FieldRecord.h"
// #include "BlockRecord.h"
// #include "MultConstRecord.h"
// #include "PenConstRecord.h"
// #include "NodeRecord.h"
// #include "SharedNodeRecord.h"
// #include "SharedNodeBuffer.h"
// #include "ExternNodeRecord.h"

// #include "Map.h"
// #include "DCRS_Matrix.h"
// #include "Vector.h"
// #include "IterativeSolver.h"
// #include "Preconditioner.h"

class ISIS_SLE : public BASE_SLE {

  public:
    //Constructor
    ISIS_SLE(MPI_Comm PASSED_COMM_WORLD, int masterRank=0);

    //Destructor
    virtual ~ISIS_SLE();

    // set parameters associated with solver choice, etc.
    void parameters(int numParams, char **paramStrings);

    void getSolnVectorPtr(Vector** vec);

    void getMatrixPtr(Matrix** mat);
    void getISISMatrixPtr(DCRS_Matrix** mat);
    void setMatrixPtr(DCRS_Matrix* mat);

    void getRHSVectorPtr(Vector** vec);
    void setNumRHSVectors(int numRHSs, int* rhsIDs);
    int setRHSID(int rhsID);
    void setRHSIndex(int index);
    void setRHSPtr(Vector* vec);

//==============================================================================
  protected:

//The following functions are associated with the specific tasks of
//using the underlying solver sub-library -- filling data structures,
//selecting and launching solver-methods, etc.
//
    void initLinearAlgebraCore();
    void deleteLinearAlgebraCore();
    void createLinearAlgebraCore(int globalNumEqns, int localStartRow,
                                 int localEndRow, int localStartCol,
                                 int localEndCol);
    void resetMatrixAndVector(double s);
    void matrixConfigure(IntArray* sysRowLengths);
    void sumIntoRHSVector(int num, const int* indices, const double* values);
    void putIntoSolnVector(int num, const int* indices, const double* values);
    double accessSolnVector(int equation);
    void sumIntoSystemMatrix(int row, int numValues, const double* values,
                             const int* scatterIndices);

    void enforceEssentialBC(int* globalEqn, double* alpha,
                                    double* gamma, int len);

    void enforceOtherBC(int* globalEqn, double* alpha, double* beta,
                        double* gamma, int len);

    void matrixLoadComplete();
    void selectSolver(char *name);
    void selectPreconditioner(char *name);
    void launchSolver(int* solveStatus);


    Map *map_;
    CommInfo *commInfo_;
    DCRS_Matrix *A_;
    DCRS_Matrix *A_ptr_;
    Dist_Vector *x_, **b_;
    Vector *b_ptr_;
    int* rhsIDs_;

    Dist_IntVector *rowLengths_;

    IterativeSolver *pSolver_;
    bool solverAllocated_;
    Preconditioner *pPrecond_;
    bool precondAllocated_;

    int internalFei_;

};

#endif

