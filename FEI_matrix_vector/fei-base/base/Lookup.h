#ifndef _Lookup_h_
#define _Lookup_h_

//
//This interface is intended to be given to a LinearSystemCore implementation,
//to allow that object to look up various information about the structure of
//the finite-element problem, map between node/field pairs and equation-numbers,
//etc.
//
//Lookup basically provides the query functions from the ProblemStructure
//class. However, this header can be included without also including other
//headers, such as intArray.h, GIDArray.h, MultConstRecord.h, and other
//headers that are required by ProblemStructure.h. ProblemStructure also
//provides a lot of functions for *setting* the structural information, which
//the LinearSystemCore object doesn't need access to.
//
//Background:
//  The finite element problem consists of a set of element-blocks, each of
//  which contains a set of elements. Each element has a list of connected
//  nodes, and each node has a set of fields. Each field consists of 1 to
//  several scalar quantities. Each of those scalar quantities corresponds
//  to an equation in the linear system. Exception: some fields may not be
//  solution-fields. This is indicated by a negative fieldID. There are no
//  equations corresponding to fields with negative fieldIDs. Data that is
//  passed in associated with negative fieldIDs is probably coordinate or
//  nullspace data, or other data passed from the application to the solver.
//
//  elem-block IDs and field IDs are application-provided numbers, and no
//  assumption may be made regarding their order, contiguousness, etc.
//
//  Equation numbers are assigned to node/field pairs by the FEI implementation,
//  and are also globally unique, and non-shared. Each equation resides on
//  exactly one processor. Equation numbers are 0-based.
//
//
//NOTES: 1. functions that return an equation number, or a size (e.g., 
//       num-equations, num-fields, etc.) may indicate an error or 'not-found'
//       condition by returning a negative number. Functions that return a
//       pointer, may indicate an error by returning NULL.
//

class Lookup {
 public:
   Lookup(){};
   virtual ~Lookup(){};

   virtual int getNumFields() = 0;

   //getFieldSize:
   //given a fieldID, this function returns the associated fieldSize

   virtual int getFieldSize(int fieldID) = 0;


   //getFieldIDsPtr:
   //returns a pointer to a list (of length numFields) of the fieldIDs

   virtual const int* getFieldIDsPtr() = 0;

   //getFieldSizesPtr:
   //returns a pointer to a list (of length numFields) of the fieldSizes

   virtual const int* getFieldSizesPtr() = 0;


   //getNumElemBlocks:
   //returns the number of element-blocks in the (local) finite-element problem.

   virtual int getNumElemBlocks() = 0;


   //getElemBlockIDs:
   //return a pointer to the list (of length numElemBlocks) containing the
   //element-block ids for the (local) finite-element problem.

   virtual const GlobalID* getElemBlockIDs() = 0;


   //getElemBlockInfo:
   //given a blockID, provide these 6 pieces of element-block information:
   //interleaveStrategy (element-equation ordering: 0 => node-major,
   //                    1 => field-major)
   //lumpingStrategy (element-matrices may be lumped if they're mass matrices,
   //                 0 => not lumped, 1 => lumped)
   //numElemDOF (number of element-dof at each element in this block)
   //numElements (number of elements in this block)
   //numNodesPerElem
   //numEqnsPerElem (number of scalar equations at each element in this block)

   virtual void getElemBlockInfo(GlobalID blockID,
                         int& interleaveStrategy, int& lumpingStrategy,
                         int& numElemDOF, int& numElements,
                         int& numNodesPerElem, int& numEqnsPerElem) = 0;


   //getNumFieldsPerNode:
   //given a blockID, return a pointer to a list (of length numNodesPerElem)
   //of numFieldsPerNode

   virtual const int* getNumFieldsPerNode(GlobalID blockID) = 0;


   //getFieldIDs
   //given a blockID, return a pointer to a table,
   // (num-rows == numNodesPerElem, row-length[i] == fieldsPerNode[i])
   //containing the fieldIDs at each node of elements in that element-block.

   virtual const int* const* getFieldIDsTable(GlobalID blockID) = 0;


   //getEqnNumber:
   //given a nodeNumber/fieldID pair, this function returns the first global
   //(0-based) equation number associated with that nodeNumber/fieldID pair.
   //!!!!! not yet implemented !!!!!

   virtual int getEqnNumber(int nodeNumber, int fieldID) = 0;


   //getNumSharedNodes:
   //return the number of local nodes that are shared by multiple processors

   virtual int getNumSharedNodes() = 0;

   //getSharedNodeNumbers:
   //return a pointer to the list of shared nodeNumbers 

   virtual const int* getSharedNodeNumbers() = 0;

   //getSharedNodeProcs:
   //given a shared nodeNumber, return a pointer to the list of sharing procs.
   //returns NULL if 'nodeNumber' is not a shared node.

   virtual const int* getSharedNodeProcs(int nodeNumber) = 0;

   //getNumSharingProcs:
   //given a shared nodeNumber, return the number of processors that share it.

   virtual int getNumSharingProcs(int nodeNumber) = 0;

   //misc point-eqn to block-eqn queries:
   //- query whether a pt-eqn corresponds exactly to a blk-eqn. in other words,
   //    is pt-eqn the first point equation in a block-equation.
   //- given a pt-eqn, return the corresponding blk-eqn
   //- given a blk-eqn and a pt-eqn, return the pt-eqn's offset into the blk-eqn
   //     (i.e., distance from the 'beginning' of the blk-eqn)
   //- given a blk-eqn, return the 'size', or number of pt-eqns corresponding
   //     to it.

   virtual bool isExactlyBlkEqn(int ptEqn) = 0;
   virtual int ptEqnToBlkEqn(int ptEqn) = 0;
   virtual int getOffsetIntoBlkEqn(int blkEqn, int ptEqn) = 0;
   virtual int getBlkEqnSize(int blkEqn) = 0;
};

#endif

