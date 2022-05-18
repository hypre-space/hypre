/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

// *************************************************************************
// Different definitions for FEI and NOFEI options
// *************************************************************************

#ifndef __HYPRE_FEI_INCLUDES__
#define __HYPRE_FEI_INCLUDES__

#ifndef NOFEI

#ifdef FEI2_1
#include "fei_defs.h"
#include "base/Data.h"
#include "base/Lookup.h"
#include "base/LinearSystemCore.h"
#else
//New FEI 2.23.02
#include "fei_defs.h"
#include "fei_Data.hpp"
#include "fei_Lookup.hpp"
#include "fei_LinearSystemCore.hpp"
#endif

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

