#ifndef __BlockRecord_H
#define __BlockRecord_H

// #include "basicTypes.h"

//------------------------------------------------------------------------------
//
//  Block is a container class for aggregating element block information
//  Normally, there will be an array of these containers, one for each
//  element block found in the FE analysis.
//
//  There is a fundamental assumption embedded in some of the methods that are
//  associated with this class, namely that the number of blocks found on a
//  given processor is small enough so that some brute-force logic (e.g.,
//  doing a linear search on a list of blocks) will not constitute a serious
//  computational burden.  If this assumption is not warranted, some of the
//  methods defined here may need some reorganization for performance gains.
//
//  kdm Sept 14, 1998
//
//------------------------------------------------------------------------------

class BlockRecord {

    public:

        BlockRecord();
        ~BlockRecord();
   
//------------------------------------------------------------------------------
//  member functions to set/get block internals

        GlobalID getBlockID();
        void setBlockID(GlobalID blockID);

        int getNumNodesPerElement();
        void setNumNodesPerElement(int numNodesPerElement);

        int getInterleaveStrategy();
        void setInterleaveStrategy(int interleaveStrategy);

        int getNumElemDOF();
        void setNumElemDOF(int numElemDOF);

        int getNumElemSets();
        void setNumElemSets(int numElemSets);

        int getNumElemTotal();
        void setNumElemTotal(int numElemTotal);
        
        int getNumEqnsPerElement();
        void setNumEqnsPerElement(int numEqnsPerElement);

        int getInitNumElemSets();
        void setInitNumElemSets(int numInitElemSets);    

        int getLoadNumElemSets();
        void setLoadNumElemSets(int numLoadElemSets);    

        int getInitNumElemTotal();
        void setInitNumElemTotal(int numInitElemTotal);    

        int getLoadNumElemTotal();
        void setLoadNumElemTotal(int numLoadElemTotal);    

        int getNumActiveNodes();
        void incrementNumActiveNodes(int numNewNodes);

        int getNumActiveEqns();
        void incrementNumActiveEqns(int numNewEqns);

        int getNextElemIndex();
        void setNextElemIndex(int nextIndex);

//------------------------------------------------------------------------------
//  member functions to set/get block list/table internals

        int *pointerToNumElemFields(int& lenNumElemFields);
        void allocateNumElemFields(int lenNumElemFields);

        int **pointerToElemFieldIDs(int& numIDRows, int* &numIDCols);
        void allocateElemFieldIDs(int numIDRows, int *numIDCols);

        GlobalID *pointerToElemIDs(int& lenElemIDs);
        void allocateElemIDs(int numElems);
	
        GlobalID** pointerToElemConn(int& numElems, int& numNodesPerElement);
        void allocateElemConn(int numElems, int numNodesPerElement);

        int *pointerToNumNodalDOF(int& lenNumNodalDOF);
        void allocateNumNodalDOF(int lenNumNodalDOF);

        int *pointerToLocalEqnElemDOF(int& numElems);
        void allocateLocalEqnElemDOF(int numElems);

        double** pointerToElemSoln(int& numElems, int& numElemDOF);
        void allocateElemSoln(int numElems, int numElemDOF);

        int findElemIndex(GlobalID elemID);

//------------------------------------------------------------------------------
//  utility member functions 

	    void dumpToScreen();
	    

    private:

//------------------------------------------------------------------------------
//  cached internals from beginInitElemBlock() and beginLoadElemBlock() methods

        GlobalID blockID_;         // ID for this block
        GlobalID *elemIDs_;        // list of element IDs for this block
        GlobalID **elemConn_;      // table of element connectivities
        double **elemSoln_;        // table of element solution parameters
        int *localEqnElemSoln_;    // list of local eqns for each elem's elemDOF
        int numNodesPerElement_;   // number of nodes for each block element
        int interleaveStrategy_;   // strategy for dense elem storage lookups
        int numElemDOF_;           // number of element soln params 
        int numElemSets_;          // current number of worksets for  block
        int numElemTotal_;         // current total number of elems in block
        int *numElemFields_;       // list of field IDs at each node in elem
        int **elemFieldIDs_;       // table of field IDs for each node
        int *numNodalDOF_;         // nodal soln cardinalities at each node

//------------------------------------------------------------------------------
//  derived internal parameters to aid in the FEI implementation

        int numEqnsPerElement_;    // number of elem eqns for each block elem
        int numInitElemSets_;      // number of init worksets for this block
        int numLoadElemSets_;      // number of load worksets for this block
        int numInitElemTotal_;     // number of init elements for this block
        int numLoadElemTotal_;     // number of load elements for this block
        int numActiveNodes_;       // number of active nodes for this block
        int numActiveEqns_;        // number of active eqns for this block
        
        int nextElemIndex_;        // index of next entry in cached elem lists

};

#endif

