#include <stdlib.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "src/BCRecord.h"
#include "src/Utils.h"

#include "src/BCManager.h"

//==============================================================================
BCManager::BCManager()
 : lenBCList_(0),
   bcList_(NULL),
   numBCNodes_(0),
   BCNodeIDs_(NULL),
   fieldsPerNode_(NULL),
   bcFieldIDs_(NULL),
   bcFieldSizes_(NULL),
   alpha_(NULL),
   beta_(NULL),
   gamma_(NULL)
{
}

//==============================================================================
BCManager::~BCManager() {
   clearAllBCs();
}

//==============================================================================
void BCManager::addBCRecord(GlobalID nodeID, int fieldID, int fieldSize,
                            const double* alpha, const double* beta,
                            const double* gamma) {

   //make a new longer list of BCRecord pointers.
   BCRecord** newBCList = new BCRecord*[lenBCList_+1];

   //copy any existing pointers into the new list.
   for(int i=0; i<lenBCList_; i++) {
      newBCList[i] = bcList_[i];
   }

   //now establish the new BCRecord pointer.
   newBCList[lenBCList_] = new BCRecord();

   BCRecord& bc = *(newBCList[lenBCList_]);

   //fill the new BCRecord with data.
   bc.setNodeID(nodeID);
   bc.setFieldID(fieldID);
   bc.setFieldSize(fieldSize);
   bc.setAlpha(alpha);
   bc.setBeta(beta);
   bc.setGamma(gamma);

   //reset the list pointer and update the length.
   delete [] bcList_;
   bcList_ = newBCList;
   lenBCList_++;
}

//==============================================================================
void BCManager::consolidateBCs() {
//
//This function takes all of the BCs that have been registered, and packs them
//down so that each node appears only once, and has a list of fields with
//boundary condition specifications.
//
   destroyAlphaBetaGamma();
   destroyFieldTables();
   destroyBCNodeIDs();

   numBCNodes_ = countBCNodes();

   setFieldTables();

   allocateAlphaBetaGamma();

   copyOverAlphaBetaGamma();
}

//==============================================================================
void BCManager::clearAllBCs() {

   destroyAlphaBetaGamma();

   destroyFieldTables();

   destroyBCNodeIDs();

   for(int j=0; j<lenBCList_; j++) {
      delete bcList_[j];
   }

   delete [] bcList_;
   bcList_ = NULL;

   lenBCList_ = 0;
}

//==============================================================================
void BCManager::destroyAlphaBetaGamma() {

   for(int i=0; i<numBCNodes_; i++) {
      for(int j=0; j<fieldsPerNode_[i]; j++) {
         delete [] alpha_[i][j];
         delete [] beta_[i][j];
         delete [] gamma_[i][j];
      }

      delete [] alpha_[i];
      delete [] beta_[i];
      delete [] gamma_[i];
   }

   delete [] alpha_;
   alpha_ = NULL;
   delete [] beta_;
   beta_ = NULL;
   delete [] gamma_;
   gamma_ = NULL;
}

//==============================================================================
void BCManager::destroyFieldTables() {
   for(int i=0; i<numBCNodes_; i++) {
      delete [] bcFieldIDs_[i];
      delete [] bcFieldSizes_[i];
   }

   delete [] bcFieldIDs_;
   bcFieldIDs_ = NULL;
   delete [] bcFieldSizes_;
   bcFieldSizes_ = NULL;

   delete [] fieldsPerNode_;
   fieldsPerNode_ = NULL;
}

//==============================================================================
void BCManager::destroyBCNodeIDs() {

   if (BCNodeIDs_ != NULL) delete [] BCNodeIDs_;
   BCNodeIDs_ = NULL;
   numBCNodes_ = 0;
}

//==============================================================================
int BCManager::countBCNodes() {
//
//This function sets the BCNodeIDs_ list, and returns the number of distinct
//BC nodes.
//
   int tmp, numNodes = 0;

   for(int i=0; i<lenBCList_; i++) {
      GlobalID nodeID = bcList_[i]->getNodeID();
      tmp = Utils::sortedGlobalIDListInsert(bcList_[i]->getNodeID(),
                                            BCNodeIDs_, numNodes);
      if(tmp<0);
   }

   return(numNodes);
}

//==============================================================================
void BCManager::setFieldTables() {
//
//Allocate the bcFieldIDs_ and bcFieldSizes_ tables, then run through the
//bcList_ setting the corresponding entries in bcFieldIDs_ and bcFieldSizes_.
//
   bcFieldIDs_ = new int*[numBCNodes_];
   bcFieldSizes_ = new int*[numBCNodes_];
   fieldsPerNode_ = new int[numBCNodes_];

   for(int i=0; i<numBCNodes_; i++) {
      bcFieldIDs_[i] = NULL;
      bcFieldSizes_[i] = NULL;
      fieldsPerNode_[i] = 0;
   }

   int ins;
   for(int j=0; j<lenBCList_; j++) {
      int index = Utils::sortedGlobalIDListFind(bcList_[j]->getNodeID(),
                                                BCNodeIDs_, numBCNodes_, &ins);

      int tmp = fieldsPerNode_[index];
      int index2 = Utils::sortedIntListInsert(bcList_[j]->getFieldID(),
                                                   bcFieldIDs_[index],
                                                   fieldsPerNode_[index]);

      int size = bcList_[j]->getFieldSize();

      if (tmp < fieldsPerNode_[index]) {
         Utils::intListInsert(size, index2, bcFieldSizes_[index], tmp);
      }
      else {
         bcFieldSizes_[index][index2] = size;
      }
   }
}

//==============================================================================
void BCManager::allocateAlphaBetaGamma() {

   alpha_ = new double**[numBCNodes_];
   beta_ = new double**[numBCNodes_];
   gamma_ = new double**[numBCNodes_];

   for(int i=0; i<numBCNodes_; i++) {
      alpha_[i] = new double*[fieldsPerNode_[i]];
      beta_[i] = new double*[fieldsPerNode_[i]];
      gamma_[i] = new double*[fieldsPerNode_[i]];

      for(int j=0; j<fieldsPerNode_[i]; j++) {
         alpha_[i][j] = new double[bcFieldSizes_[i][j]];
         beta_[i][j] = new double[bcFieldSizes_[i][j]];
         gamma_[i][j] = new double[bcFieldSizes_[i][j]];

         for(int k=0; k<bcFieldSizes_[i][j]; k++) {
            alpha_[i][j][k] = 0.0;
            beta_[i][j][k] = 0.0;
            gamma_[i][j][k] = 0.0;
         }
      }
   }
}

//==============================================================================
void BCManager::copyOverAlphaBetaGamma() {
//
//This function copies the BC specifications out of the bcList_ into the
//consolidated alpha/beta/gamma arrays.
//It's done such that essential BCs are "last one in wins" -- i.e., essential
//BCs overwrite any BC that may already exist at a field for a given node.
//Natural or mixed BCs are summed into any existing BCs, but not if there's
//already an essential BC in place.
//
//  here's a natural place to check for any conflicting BCs and take any
//  corrective actions that may be possible
//
   int tmp;
   for(int i=0; i<lenBCList_; i++) {
      int ind1 = Utils::sortedGlobalIDListFind(bcList_[i]->getNodeID(),
                                               BCNodeIDs_, numBCNodes_, &tmp);

      int ind2 = Utils::sortedIntListFind(bcList_[i]->getFieldID(),
                                          bcFieldIDs_[ind1],
                                          fieldsPerNode_[ind1], &tmp);

      double* alpha = bcList_[i]->pointerToAlpha();
      double* beta = bcList_[i]->pointerToBeta();
      double* gamma = bcList_[i]->pointerToGamma();

      int size = bcFieldSizes_[ind1][ind2];

      double* alphaPtr = alpha_[ind1][ind2];
      double* betaPtr = beta_[ind1][ind2];
      double* gammaPtr = gamma_[ind1][ind2];

      for(int k=0; k<size; k++) {
         if ((alpha[k] != 0.0) && (beta[k] == 0.0)) {
            //it's essential, so 'put' it in
            alphaPtr[k] = alpha[k];
            betaPtr[k] = beta[k];
            gammaPtr[k] = gamma[k];
         }
         else {
            //it's natural or mixed, so sum it in, but not if
            //an essential BC has already been put in.
            if (!((alphaPtr[k] != 0.0) && (betaPtr[k] == 0.0))){
               alphaPtr[k] += alpha[k];
               betaPtr[k] += beta[k];
               gammaPtr[k] += gamma[k];
            }
         }
      }

      //at this point we're done with bcList_[i].
      delete bcList_[i];
   }

   delete [] bcList_;
   bcList_ = NULL;
   lenBCList_ = 0;
}

