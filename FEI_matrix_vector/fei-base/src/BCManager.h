#ifndef _BCManager_h_
#define _BCManager_h_

//requires:
//#include "other/basicTypes.h"
//#include "src/BCRecord.h"

class BCManager {
 public:
   BCManager();
   virtual ~BCManager();

   void addBCRecord(GlobalID nodeID, int fieldID, int fieldSize,
                    const double* alpha, const double* beta,
                    const double* gamma);

   void consolidateBCs();

   int getNumBCNodes() {return(numBCNodes_);};
   GlobalID* getBCNodeIDsPtr() {return(BCNodeIDs_);};

   int* fieldsPerNodePtr() {return(fieldsPerNode_);};

   int** bcFieldIDsPtr() {return(bcFieldIDs_);};
   int** bcFieldSizesPtr() {return(bcFieldSizes_);};

   double*** alphaPtr() {return(alpha_);};
   double*** betaPtr() {return(beta_);};
   double*** gammaPtr() {return(gamma_);};

   void clearAllBCs();

 private:
   int countBCNodes();
   void setFieldTables();
   void allocateAlphaBetaGamma();
   void copyOverAlphaBetaGamma();
   void destroyAlphaBetaGamma();
   void destroyFieldTables();
   void destroyBCNodeIDs();

   int lenBCList_;
   BCRecord** bcList_;

   int numBCNodes_;      //how many nodes have boundary conditions specified
   GlobalID* BCNodeIDs_; //the IDs of the BC nodes
   int* fieldsPerNode_;  //number of fields on which each node has BCs specified
   int** bcFieldIDs_;    //table with 'numBCNodes_' rows, and row i is of
                         //length 'fieldsPerNode_[i]'.
   int** bcFieldSizes_;  //same dimensions as bcFieldIDs_

   double*** alpha_;  //list of tables. number-of-tables = numBCNodes_
                      //table i has number-of-rows = fieldsPerNode_[i], and
                      //row j of table i has length bcFieldSizes_[i][j].
   double*** beta_;
   double*** gamma_;
};

#endif

