//
//This is a class that will exercise FEI implementations.
//
//This class sets up test data simulating a 2D bar, and uses
//that data in calling the FEI functions that make up the
//initializaztion phase and the load phase.
//
// This program will not really solve the finite element problem.
// The structure of the problem will be used to assemble a matrix through
// the FEI, but the actual (stiffness, etc.) data supplied will not be
// very meaningful. However, the program is stuctured such that meaningful
// data may be added easily later.
//
//The calling program (the 'user' of PoissonData) is left
//with the task of calling the other 'easier' FEI functions
//such as setting matrix contexts, resetSystem(...), iterateToSolve(),
//parameters(...), etc.
//
//******************************************************************
//Note that the calling program should construct the FEI instance
//AND call initSolveStep, setting solveType, numMatrices, etc., BEFORE
//calling PoissonData's do_init_phase or do_load_phase functions.
//******************************************************************
//
//Also note:
//
// 1. This class only uses 1 element-block per processor currently.
// 2. This class only uses 1 workset per block currently.
// 3. There are mutually exclusive load_phase alternatives.
//    Either call do_load_phase(), which does all data loading,
//    OR
//    call
//         do_load_BC_phase(),
//
//         do_begin_load_block(),
//            do_load_stiffnes(),
//            do_load_rhs(),
//         do_end_load_block(),
//
//         do_loadComplete().
//
// 4. do_load_BC_phase() can be called either before or after the
// begin/end data loading block, but must be called before loadComplete().
//
// Alan Williams 6-07-99
//

class PoissonData {
  public:
    //constructor -- see PoissonData.cpp for descriptions of these
    //parameters.
    PoissonData(int L, int W, int DOF,
               int numProcs, int localProc, int outputLevel);

    //destructor.
    ~PoissonData();

    void print_storage_summary();

    void set_FEI_instance(FEI& fei);

    void do_init_phase();

    //for loading......................
    void do_load_phase();

    //OR, for alternative loading......
    void do_load_BC_phase();

    void do_begin_load_block();

    void do_load_stiffness();
    void do_load_rhs();

    void do_end_load_block();

    void do_loadComplete();
    //.................................

    GlobalID getLocalBlockID() {return(elemBlockID_);};

  private:
    void check1();
    void calculateDistribution();

    void messageAbort(char* message);

    void initializeElements();
    void calculateConnectivity(GlobalID* conn, int size, GlobalID elemID);
    void initializeFieldStuff();
    void deleteFieldArrays();

    void allocateSharedNodeArrays();
    void deleteSharedNodeArrays();
    void getLeftSharedNodes(int& numShared);
    void getRightSharedNodes(int& numShared);
    void getTopSharedNodes(int& numShared);
    void getBottomSharedNodes(int& numShared);
    void printSharedNodes(char* str, int numShared, GlobalID* nodeIDs,
                          int** shareProcs, int* numShareProcs);

    void calculateBCs();
    void deleteBCArrays();
    void addBCNode(GlobalID nodeID, double x, double y);
    int appendBCNodeID(GlobalID nodeID);
    void appendBCRow(double**& valTable, double value);
    void printBCs();

    Poisson_Elem* elems_;
    bool elemsAllocated_;
    int numLocalElements_;
    int startElement_;

    int numProcs_;
    int localProc_;
    int outputLevel_;

    int L_, W_, DOF_;
    int procX_, procY_;
    int maxProcX_, maxProcY_;

    FEI* fei_;
    bool fei_has_been_set;

    int numElemBlocks_;
    int solveType_;

    int nodesPerElement_;
    int fieldsPerNode_;
    GlobalID elemBlockID_;
    int elemSetID_;
    int elemFormat_;

    //*************** field description variables *********
    int fieldSize_;
    int* numFields_;
    int** fieldIDs_;
    bool fieldArraysAllocated_;

    //************* element IDs and connectivities ********
    GlobalID* elemIDs_;
    bool elemIDsAllocated_;
    GlobalID** elemConn_;
    bool elemConnAllocated_;
    double*** elemStiff_;
    bool elemStiffAllocated_;
    double** elemLoad_;
    bool elemLoadAllocated_;

    //************* shared node arrays ********************
    int numSharedNodeSets_;
    GlobalID* sharedNodeIDs_;
    bool sharedNodeIDsAllocated_;
    int numSharedNodes_;
    int procsPerNode_;
    int** sharedNodeProcs_;
    bool sharedNodeProcsAllocated_;
    int* numSharedNodeProcs_;

    //************* boundary condition stuff **************
    int numBCNodeSets_;
    GlobalID* BCNodeIDs_;
    bool BCNodeIDsAllocated_;
    int numBCNodes_;
    double** alpha_;
    double** beta_;
    double** gamma_;
    bool BCArraysAllocated_;

};

