#ifndef _CommBufferInt_h_
#define _CommBufferInt_h_

//
// This class (and the CommBufferDouble class) is a buffering class in which
// doubles can be progressively accumulated. These ints are intended to
// be indices (from a row of a matrix), and so are associated with an
// equation number (which will be the matrix-row in which they belong).
//
// When all indices have been buffered, the user can call
// the buildLists function, which marshalls the information into
// tables, one row for each destination processor. Pointers to the rows
// of these tables can then be obtained for passing as MPI messages.
//
// For each equation-numbers list that will be produced, there will be
// an offsets list (of length numEqns+1, so that the number of indices for
// each eqn is given by offsets[i+1]-offsets[i]), which gives the offset
// into the corresponding indices list at which that equation's
// data begins.
//
// Example usage:
//
// **************** Sending processor: *******************
//
//    CommBufferDouble commBufD;
//    CommBufferInt commBufI;
//
//    for(loop over a bunch of equations 'eqn') {
//       commBufD.addDoubles(eqn, coefficients, numCoefficients, procNum);
//       commBufI.addInts(eqn, colInds, numIndices, procNum);
//    }
//
//   commBufD.buildLists();
//   commBufI.buildLists();
//
//
//   //now we want to send stuff to processor 'proc':
//
//   int numEqns, numCoefs, numIndices;
//   int* eqnList = commBufD.pointerToEqnNumbers(numEqns, proc);
//   double* coefs = commBufD.pointerToDoubles(numCoefs, proc);
//   int* indices = commBufI.pointerToInts(numIndices, proc);
//   int* offsets = commBufD.pointerToOffsets(numOffsets, proc);
//
//   MPI_Send(numEqns, 1, ...);
//   MPI_Send(numCoefs, 1, ...);
//
//   MPI_Send(eqnList, numEqns, ...);
//   MPI_Send(coefs, numCoefs, ...);
//   MPI_Send(indices, numIndices, ...);
//   MPI_Send(offsets, numEqns, ...);
//
// ************* Recv'ing processor: *******************
//
//   //recv the length parameters numEqns and numCoefs, and
//   //allocate the arrays to recv into... then:
//
//   MPI_Recv(eqnNums, ...
//   MPI_Recv(coefs, ...
//   MPI_Recv(indices, ...
//   MPI_Recv(offsets, ...
//
//   for(i=0; i<numEqns; i++) {
//      //example: summing into a matrix row, where the arguments
//      //are: sumIntoRow(int row, int numCoefs, double* coefs, int* indices);
//
//      A_->sumIntoRow(eqnNums[i], offsets[i+1]-offsets[i],
//                     &(coefs[offsets[i]]), &(indices[offsets[i]]) );
//   }
// 

class CommBufferInt {

  public:

    CommBufferInt();
    virtual ~CommBufferInt();

    void addInts(int eqnNum, const int* indices, int numIndices, int destProc);

    void buildLists();

    int* pointerToEqnNumbers(int& numEqnNumbers, int destProc);

    int* pointerToInts(int& numInts, int destProc);

    int* pointerToOffsets(int& numOffsets, int destProc);

    int* pointerToProcs(int& numProcs);

  private:

    void initializeTables();
    void destroyTables();
    void buildProcList();
    void buildTables();

    IntArray* eqnNumbers_;      //table of equation numbers.

    IntArray* ints_;        //table of coefficients.

    IntArray* offsets_;         //table of offsets.

    IntArray* destProcs_;       //list of (unique) destination processors.
    int numDestProcs_;          //number-of-rows for the above tables.

    //====================================================================
    //Following is 'temp' data that will grow as the add* functions are
    //called, and from which the above tables/lists will be built when
    //the buildLists() function is called.

    IntArray* addEqnNumbers_;
    IntArray** addInts_;
    IntArray* addDestProcs_;

};

#endif

