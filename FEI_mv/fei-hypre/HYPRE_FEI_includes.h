/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

// *************************************************************************
// Different definitions for FEI and NOFEI options
// *************************************************************************

#ifndef __HYPRE_FEI_INCLUDES__
#define __HYPRE_FEI_INCLUDES__

#ifndef NOFEI

#include "fei_defs.h"
#include "Data.h"
#include "Lookup.h"
#include "LinearSystemCore.h"

#else

#define GlobalID int
class Lookup { 
public:
   Lookup() {}
   ~Lookup() {}
   int  getNumFields() {return -1;}
   int  getFieldSize(int) {return -1;}
   int* getFieldIDsPtr() {return NULL;}
   int* getFieldSizesPtr() {return NULL;}
   int  getNumElemBlocks() {return -1;}
   int* getElemBlockIDs() {return NULL;}
   void getElemBlockInfo(int,int&,int&,int&,int&,int&,int&){return;}
   int *getNumFieldsPerNode(int) {return NULL;}
   int**getFieldIDsTable(int) {return NULL;}
   int  getEqnNumber(int, int) {return -1;}
   int  getAssociatedNodeNumber(int) {return -1;}
   int  getAssociatedFieldID(int) {return -1;}
   int  isInLocalElement(int) {return -1;}
   int  getNumSubdomains(int) {return -1;}
   int *getSubdomainList(int) {return NULL;}
   int  getNumSharedNodes() {return -1;}
   int *getSharedNodeNumbers() {return NULL;}
   int *getSharedNodeProcs(int) {return NULL;}
   int  getNumSharingProcs(int) {return -1;}
   int  isExactlyBlkEqn(int) {return -1;}
   int  ptEqnToBlkEqn(int) {return -1;}
   int  getOffsetIntoBlkEqn(int, int) {return -1;}
   int  getBlkEqnSize(int) {return -1;}
};

#endif

#endif

