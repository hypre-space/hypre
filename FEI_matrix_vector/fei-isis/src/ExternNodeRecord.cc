#include <stdio.h>
#include <stdlib.h>
#include <iostream.h>
#include <assert.h>

#include "other/basicTypes.h"
//#include "mv/IntArray.h"
//#include "mv/GlobalIDArray.h"
#include "isis-mv/IntArray.h"
#include "isis-mv/GlobalIDArray.h"
#include "src/SLE_utils.h"
#include "src/ExternNodeRecord.h"

//====================================================================
ExternNodeRecord::ExternNodeRecord() {
//
//  Constructor. Initialize member stuff.
//

    allExtNodes_ = NULL;
    numAllExtNodes_ = 0;

    remoteNodes_ = NULL;
    numRemoteNodes_ = 0;

    localNodes_ = NULL;
    numLocalNodes_ = 0;
    externNodeList_ = NULL;

    recvProcList_ = NULL;
    lenRecvProcList_ = 0;
    sendProcList_ = NULL;
    lenSendProcList_ = 0;

    localNodeIDs_ = NULL;
    DOFs_ = NULL;             //$kdm - is this used anywhere?
    lenRecv_ = NULL;
    remoteNodeIDs_ = NULL;
    lenSend_ = NULL;

    localNumPenCRs_ = 0;
    locPenNodes_ = NULL;
    lenLocPenNodes_ = NULL;
    locPenEqnIDs_ = NULL;
    locPenNumDOF_ = NULL;
    locPenID_ = NULL;
    locPenProcs_ = NULL;
    lenLocPenProcs_ = NULL;

    remoteNumPenCRs_ = 0;
    remPenNodes_ = NULL;
    lenRemPenNodes_ = NULL;
    remPenEqnIDs_ = NULL;
    remPenNumDOF_ = NULL;
    remPenID_ = NULL;
    remPenProcs_ = NULL;

    return;
}

//====================================================================
ExternNodeRecord::~ExternNodeRecord() {
//
//  Destructor. Delete all allocated memory.
//
    int i;

    delete [] externNodeList_;

    if (allExtNodes_ != NULL) {
        for(i = 0; i < numAllExtNodes_; i++) {
            deleteExtNode(allExtNodes_[i]);
        }
        delete [] allExtNodes_;
    }

    if (localNodes_ != NULL) {
        for(i = 0; i < numLocalNodes_; i++) {
            deleteExtNode(localNodes_[i]);
        }
        delete [] localNodes_;
    }

    if (remoteNodes_ != NULL) {
        for(i = 0; i < numRemoteNodes_; i++) {
            deleteExtNode(remoteNodes_[i]);
        }
        delete [] remoteNodes_;
    }

    if (remoteNodeIDs_ != NULL) {
        for(i = 0; i < lenRecvProcList_; i++) {
            delete [] remoteNodeIDs_[i];
        }
        delete [] remoteNodeIDs_;
    }

    if (recvProcList_ != NULL) {
        delete [] recvProcList_;
    }    
    if (lenRecv_ != NULL) {
        delete [] lenRecv_;
    }

    if (localNodeIDs_ != NULL) {
        for(i = 0; i < lenSendProcList_; i++) {
            delete [] localNodeIDs_[i];
        }
        delete [] localNodeIDs_;
    }
    
    if (sendProcList_ != NULL) {
        delete [] sendProcList_;
    }
    if (lenSend_ != NULL) {
        delete [] lenSend_;
    }

    if (locPenNodes_ != NULL) {
        for(i = 0; i < localNumPenCRs_; i++) {
            delete [] locPenNodes_[i];
        }
        delete [] locPenNodes_;
    }

    if (locPenEqnIDs_ != NULL) {
        for(i = 0; i < localNumPenCRs_; i++) {
            delete [] locPenEqnIDs_[i];
        }
        delete [] locPenEqnIDs_;
    }

    if (locPenNumDOF_ != NULL) {
        for(i = 0; i < localNumPenCRs_; i++) {
            delete [] locPenNumDOF_[i];
        }
        delete [] locPenNumDOF_;
    }
 
    if (locPenProcs_ != NULL) {
        for(i = 0; i < localNumPenCRs_; i++) {
            delete [] locPenProcs_[i];        
        }
        delete [] locPenProcs_;
    }

    if (locPenID_ != NULL) {
        delete [] locPenID_;
    }
    if (lenLocPenNodes_ != NULL) {
        delete [] lenLocPenNodes_;
    }
    if (lenLocPenProcs_ != NULL) {
        delete [] lenLocPenProcs_;
    }

    if (remPenNodes_ != NULL) {
        for(i = 0; i < remoteNumPenCRs_; i++) {
            delete [] remPenNodes_;
        }
        delete [] remPenNodes_;
    }

    if (remPenEqnIDs_ != NULL) {
        for(i = 0; i < remoteNumPenCRs_; i++) {
            delete [] remPenEqnIDs_;
        }
        delete [] remPenEqnIDs_;
    }

    if (remPenNumDOF_ != NULL) {
        for(i = 0; i < remoteNumPenCRs_; i++) {
            delete [] remPenNumDOF_;
        }
        delete [] remPenNumDOF_;
    }

    if (lenRemPenNodes_ != NULL) {
        delete [] lenRemPenNodes_;
    }
    if (remPenID_ != NULL) {
        delete [] remPenID_;
    }
    if (remPenProcs_ != NULL) {
        delete [] remPenProcs_;
    }

    return;
}

//====================================================================
ExtNode* ExternNodeRecord::newExtNode(){
//
//  'new' into existence, an ExtNode struct.
//
    ExtNode *newNode = new ExtNode;

    newNode->nodeID = (GlobalID)(-1);
    newNode->ownerProc = -1;
    newNode->procIDs = NULL;
    newNode->numProcs = 0;
    newNode->numSolnParams = 0;
    newNode->globalEqn = -1;
    newNode->numMultCRs = 0;
    newNode->numPenCRs = 0;
    newNode->globalCREqn = NULL;
    newNode->penaltyTerm = false;

    newNode->numFields = 0;
    newNode->fieldIDs = NULL;
    newNode->fieldOffsets = NULL;

    return(newNode);
}

//====================================================================
void ExternNodeRecord::deleteExtNode(ExtNode *node){
//
//  Simply deletes the memory associated with an ExtNode struct.
//
    
    delete [] node->procIDs;
    delete [] node->globalCREqn;
    delete [] node->fieldIDs;
    delete [] node->fieldOffsets;
    delete node;

    return;
}

//====================================================================
void ExternNodeRecord::externNodes(GlobalID *nodeIDs, int numNodes,
                     int **procs, int *lenProcs){
//
//  This function is used for passing in the parameters that we get from
//  the FEI functions initExtSendNodes and initExtRecvNodes.
//
//  This is where the array allExtNodes_ will be established or added
//  to.
//
    int i, j, index;
    ExtNode **newPtr = NULL;

    if (numAllExtNodes_ <= 0) {
        //We are now establishing the array allExtNodes_. i.e., these
        //incoming nodes are the first ones we've been given.

        numAllExtNodes_ = numNodes;
        allExtNodes_ = new ExtNode*[numNodes];

        //allocate numNodes new ExtNode's.
        for(i=0; i<numNodes; i++) {
            allExtNodes_[i] = newExtNode();
            allExtNodes_[i]->nodeID = nodeIDs[i];
            procIDs(allExtNodes_[i], procs[i], lenProcs[i]);
        }
    }
    else {
        //numAllExtNodes_ > 0, so we will be adding to the existing
        //array of external nodes.
        for(i=0; i<numNodes; i++) {
            if ((index = allExtNodeIndex(nodeIDs[i])) >= 0) {
                //This node is already here, so just add these procs to its
                //record.
                procIDs(allExtNodes_[index], procs[i], lenProcs[i]);
            }
            else {
                //This is a new node, so we'll have to lengthen the array
                //of external nodes and add it in.

                newPtr = new ExtNode*[numAllExtNodes_+1];
                
                //copy the existing pointers into the new array
                for(j=0; j<numAllExtNodes_; j++) {
                    newPtr[j] = allExtNodes_[j];
                }

                //now put in the new node. all we know about the new node
                //so far is its nodeID and its procs info.
                newPtr[numAllExtNodes_] = newExtNode();

                newPtr[numAllExtNodes_]->nodeID = nodeIDs[i];
                procIDs(newPtr[numAllExtNodes_], procs[i], lenProcs[i]);

                //now tidy up.
                delete [] allExtNodes_;
                allExtNodes_ = newPtr;

                numAllExtNodes_++;
            }
        }
    }

    return;
}

//====================================================================
void ExternNodeRecord::initComplete() {
//
//  This function is called after the remoteNodes and corresponding
//  information (such as numMultCRs, numPenCRs and globalCREqn for recv nodes)
//  have been loaded in. This function collects that information into linear
//  lists that can be easily passed off to the other processors.
//
//  This function assumes that the numMultCRs and numPenCRs variables have
//  already been set, since this is how it tells the difference between the
//  local and remote nodes. We're making an assumption here: if a node has
//  been passed in as an external node, but has not been flagged as appearing
//  in any local constraint relations, then we assume that it is a local
//  node.
//
    int i,j,proc,pIndex;

    //first, we're going to build the recvProcList_ and sendProcList_
    //arrays.
    for(i=0; i<numAllExtNodes_; i++){
        if ((allExtNodes_[i]->numMultCRs > 0) || 
            (allExtNodes_[i]->numPenCRs > 0)) {
            //if numMultCRs>0 or numPenCRs>0 then this is probably a remote
            //node. i.e., it appears in one or more CRs on this processor, and
            //we'll be recv'ing its globalEqn and DOF info from its owning
            //processor. We'll also be telling the owning processor how many
            //CRs it appears in here.

            for(j=0; j<allExtNodes_[i]->numProcs; j++){
                proc = allExtNodes_[i]->procIDs[j];

                //now append this processor to the recv proc list if it
                //isn't already in there.
                if (!inList(recvProcList_, lenRecvProcList_, proc)) {
                    appendIntList(&recvProcList_, &lenRecvProcList_, proc);
                }
            }
        }
        else {
            //since neither numMultCRs nor numPenCRs is >0, this is a local 
            //node. i.e., it appears in 
            //a CR on another processor, and we need to send its globalEqn
            //and DOF information. The remote processor will also be telling
            //us the numMultCRs and globalCREqn info for this node, as well as
            //the lists of other nodes in the penalty constraint, if any.

            for(j=0; j<allExtNodes_[i]->numProcs; j++){
                proc = allExtNodes_[i]->procIDs[j];

                //now append this processor to the send proc list if it
                //isn't already in there.
                if (getSendProcIndex(proc) < 0) {
                    appendIntList(&sendProcList_, &lenSendProcList_, proc);
                }
            }
        }
    }

    //now, we can build the remoteNodeIDs_ and localNodeIDs_ arrays.

    if (lenRecvProcList_ > 0) {
        remoteNodeIDs_ = new GlobalID*[lenRecvProcList_];
        lenRecv_ = new int[lenRecvProcList_];
        for(i=0; i<lenRecvProcList_; i++) {
            remoteNodeIDs_[i] = NULL;
            lenRecv_[i] = 0;
        }
    }

    if (lenSendProcList_ > 0) {
        localNodeIDs_ = new GlobalID*[lenSendProcList_];
        lenSend_ = new int[lenSendProcList_];
        for(i=0; i<lenSendProcList_; i++) {
            localNodeIDs_[i] = NULL;
            lenSend_[i] = 0;
        }
    }

    for(i=0; i<numAllExtNodes_; i++){
        if ((allExtNodes_[i]->numMultCRs >0) ||(allExtNodes_[i]->numPenCRs>0)){
            //same test as above, i.e., if numCRs>0, it's a remote node.

            for(j=0; j<allExtNodes_[i]->numProcs; j++){
                //first get the index into the recvProcList_ array which
                //will put us in the right row of remoteNodeIDs_.
                pIndex = getRecvProcIndex(allExtNodes_[i]->procIDs[j]);
                if (pIndex<0) {
                    cout << "ExternNodeRecord::initComplete: ERROR, pIndex<0"
                         << endl;
                    exit(0);
                }

                if (lenRecv_[pIndex] == 0) {
                    remoteNodeIDs_[pIndex] = new GlobalID[1];
                    remoteNodeIDs_[pIndex][0] = allExtNodes_[i]->nodeID;
                    lenRecv_[pIndex] = 1;
                }
                else {
                    //***NOTE: I'm using the lenRecv_ array to keep track of the
                    //length of the rows of the remoteNodeIDs_ array.
                    appendGlobalIDList(&(remoteNodeIDs_[pIndex]), 
                                  &(lenRecv_[pIndex]), allExtNodes_[i]->nodeID);
                }
            }
        }
        else {
            //numCRs<=0, so this is a local node.

            for(j=0; j<allExtNodes_[i]->numProcs; j++){
                //first get the index into the sendProcList_ array which
                //will put us in the right row of localNodeIDs_.
                pIndex = getSendProcIndex(allExtNodes_[i]->procIDs[j]);
                if (pIndex<0) {
                    cout << "ExternNodeRecord::initComplete: ERROR, pIndex<0"
                         << endl;
                    exit(0);
                }

                appendGlobalIDList(&(localNodeIDs_[pIndex]),
                                   &(lenSend_[pIndex]),
                                   allExtNodes_[i]->nodeID);
            }
        }
    }

    return;
}

//====================================================================
void ExternNodeRecord::localPenNodeList(int penID, GlobalID *nodeList,
                                   int lenList){
//
//  This function adds a list of nodes associated with a penalty
//  constraint node, to a table of nodes.
//  This function also allocates the corresponding table of penEqnIDs
//  and penNumDOFs, and list of penIDs.
//  This is also where we update the localNumPenCRs variable.
//

    if (getLocPenIDIndex(penID) < 0){
        //the table doesn't already contain this penID, so append the new list

        int tmp = localNumPenCRs_;
        appendGlobalIDTableRow(&locPenNodes_, &lenLocPenNodes_, tmp,
                               nodeList, lenList);

        tmp = localNumPenCRs_;
        appendIntTableRow(&locPenEqnIDs_, &lenLocPenNodes_, tmp,
                          NULL, lenList);

        tmp = localNumPenCRs_;
        appendIntTableRow(&locPenNumDOF_, &lenLocPenNodes_, tmp,
                          NULL, lenList);

        tmp = localNumPenCRs_;
        appendIntList(&locPenID_, &tmp, penID);

        localNumPenCRs_++;
    }

    return;  
}

//====================================================================
void ExternNodeRecord::remotePenNodeList(int penID, int proc, 
                                         GlobalID *nodeList, int lenList){
//
//  This function adds a list of nodes associated with a penalty
//  constraint node, to a table of nodes.
//  This function also allocates the corresponding table of penEqnIDs
//  and penNumDOFs, and list of penIDs.
//  This is also where we update the remoteNumPenCRs variable.
//

    int index = getRemotePenIDIndex(penID,proc);

    if (index < 0) {
        //the table doesn't already contain this penID, at least not from
        //the same processor, so append the new list

        int tmp = remoteNumPenCRs_;
        appendGlobalIDTableRow(&remPenNodes_, &lenRemPenNodes_, tmp,
                               nodeList, lenList);

        tmp = remoteNumPenCRs_;
        appendIntTableRow(&remPenEqnIDs_, &lenRemPenNodes_, tmp,
                          NULL, lenList);

        tmp = remoteNumPenCRs_;
        appendIntTableRow(&remPenNumDOF_, &lenRemPenNodes_, tmp,
                          NULL, lenList);

        tmp = remoteNumPenCRs_;
        appendIntList(&remPenID_, &tmp, penID);

        tmp = remoteNumPenCRs_;
        appendIntList(&remPenProcs_, &tmp, proc);

        remoteNumPenCRs_++;
    }

    return;
}
 
//====================================================================
void ExternNodeRecord::localPenEqnIDList(int penID, int *eqnIDList,
                                         int lenList){
//
//  This function is for putting in the list of eqnIDs for penalty 
//  constraint penID 
//
//  This function can not be called until after the penNodes list for penID
//  has been put in using the penNodeList function.
//
    int penIDIndex = getLocPenIDIndex(penID);
    int i;
    
    //make sure the list is the right length
    if (lenLocPenNodes_[penIDIndex] != lenList){
        cout << "ERROR, ExternNodeRecord::localPenEqnIDList: list wrong length"
            << endl << flush;
        exit(0);
    }

    //now copy it into the correct row of the table.
    for(i=0; i<lenList; i++){
        locPenEqnIDs_[penIDIndex][i] = eqnIDList[i];
    }

    return;
}

//====================================================================
void ExternNodeRecord::remotePenEqnIDList(int penID, int proc,
                                          int *eqnIDList, int lenList){
//
//  This function is for putting in the list of eqnIDs for penalty constraint
//  penID
//
//  This function can not be called until after the penNodes list for penID
//  has been put in using the penNodeList function.
//
    int penIDIndex = getRemotePenIDIndex(penID, proc);
    int i;

    if (penIDIndex >= 0){
        //make sure the list is the right length
        if (lenRemPenNodes_[penIDIndex] != lenList){
            cout << "ERROR, ExternNodeRecord::remotePenEqnIDList: wrong length"
                << endl << flush;
            exit(0);
        }

        //now copy it into the correct row of the table.
        for(i=0; i<lenList; i++){
            remPenEqnIDs_[penIDIndex][i] = eqnIDList[i];
        }
    }

    return;
}
 
//====================================================================
void ExternNodeRecord::localPenNumDOFList(int penID, int *DOFList,
                                          int lenList){
//
//  This function is for putting the list of DOFs for penalty constraint
//  penID, into the table of DOFs.
//
//  This function can not be called until after the penNodes list
//  has been put in using the penNodeList function.
//
    int penIDIndex = getLocPenIDIndex(penID);
    int i;

    if (penIDIndex >= 0) {
        //make sure the list is the right length
        if (lenLocPenNodes_[penIDIndex] != lenList){
            cout << "ERROR, ExternNodeRecord::locPenNumDOFList: wrong length"
                << endl << flush;
            exit(0);
        }

        //now copy it into the correct row of the table.
        for(i=0; i<lenList; i++){
            locPenNumDOF_[penIDIndex][i] = DOFList[i];
        }
    }

    return;
}
 
//====================================================================
void ExternNodeRecord::remotePenNumDOFList(int penID, int proc, int *DOFList,
                                          int lenList){
//
//  This function is for putting the list of DOFs for penalty constraint
//  penID, into the table of DOFs.
//
//  This function can not be called until after the penNodes list
//  has been put in using the penNodeList function.
//
    int penIDIndex = getRemotePenIDIndex(penID, proc);
    int i;

    if (penIDIndex >= 0) {
        //make sure the list is the right length
        if (lenRemPenNodes_[penIDIndex] != lenList){
            cout << "ERROR, ExternNodeRecord::remPenNumDOFList: wrong length"
                 << endl << flush;
            exit(0);
        }

        //now copy it into the correct row of the table.
        for(i=0; i<lenList; i++){
            remPenNumDOF_[penIDIndex][i] = DOFList[i];
        }
    }

    return;
}
 
//====================================================================
int ExternNodeRecord::remotePenArraySize(int penID, int proc){
//
// This function calculates the array size needed to catch an incoming
// message which contains the weights, a CRValue and a PenValue for the
// remote penalty constraint penID from processor proc.
//
    int i;
    int index = getRemotePenIDIndex(penID, proc);
    if (index<0){
        cout << "ERROR, ExternNodeRecord::remotePenArraySize, index<0"
             << endl << flush;
        exit(0);
    }

    int size = 0;

    //first, the number of weights...
    for(i=0; i<lenRemPenNodes_[index]; i++){
        size += remPenNumDOF_[index][i];
    }

    //plus, a CRValue and a penValue...
    size += 2;

    return(size);
}

//====================================================================
GlobalID **ExternNodeRecord::localPenNodeTable(int **lenPenNodes){

    *lenPenNodes = lenLocPenNodes_;
    return(locPenNodes_);
}

//====================================================================
GlobalID **ExternNodeRecord::remotePenNodeTable(int **lenPenNodes){

    *lenPenNodes = lenRemPenNodes_;
    return(remPenNodes_);
}
 
//====================================================================
int **ExternNodeRecord::localPenEqnIDTable(int **lenPenNodes){

    *lenPenNodes = lenLocPenNodes_;
    return(locPenEqnIDs_);
}
 
//====================================================================
int **ExternNodeRecord::remotePenEqnIDTable(int **lenPenNodes){

    *lenPenNodes = lenRemPenNodes_;
    return(remPenEqnIDs_);
}
 
//====================================================================
int **ExternNodeRecord::localPenNumDOFTable(int **lenPenNodes){

    *lenPenNodes = lenLocPenNodes_;
    return(locPenNumDOF_);
}
 
//====================================================================
int **ExternNodeRecord::remotePenNumDOFTable(int **lenPenNodes){

    *lenPenNodes = lenRemPenNodes_;
    return(remPenNumDOF_);
}
 
//====================================================================
int* ExternNodeRecord::localPenIDList(int& lclNumPenCRs){
    lclNumPenCRs = localNumPenCRs_;
    return(locPenID_);
}

//====================================================================
int* ExternNodeRecord::remotePenIDList(int& remNumPenCRs){
    remNumPenCRs = remoteNumPenCRs_;
    return(remPenID_);
}
 
//====================================================================
void ExternNodeRecord::calcPenProcTable(){
//
//  This function creates and fills the locPenProcs_ table.
//  The number of rows in this table is the number of local penalty constraints
//  which contain external nodes. Each row of the table contains a list of
//  the processors which the penalty stuff will be sent to.
//
    int i, j;
    if (locPenProcs_ != NULL) {
        for(i=0; i<localNumPenCRs_; i++){
            delete [] locPenProcs_[i];
        }
        delete [] locPenProcs_;
        delete [] lenLocPenProcs_;
    }

    locPenProcs_ = new int*[localNumPenCRs_];
    lenLocPenProcs_ = new int[localNumPenCRs_];

    for(i=0; i<localNumPenCRs_; i++){
        locPenProcs_[i] = NULL;
        lenLocPenProcs_[i] = 0;

        for(j=0; j<lenLocPenNodes_[i]; j++){
            int proc = ownerProc(locPenNodes_[i][j]);

            if (!inList(locPenProcs_[i], lenLocPenProcs_[i], proc) &&
                (proc >= 0)){
                appendIntList(&locPenProcs_[i], &lenLocPenProcs_[i], proc);
            }
        }
    }

    return;
}

//====================================================================
int** ExternNodeRecord::localPenProcTable(int **lenProcTable){
    *lenProcTable = lenLocPenProcs_;
    return(locPenProcs_);
}

//====================================================================
int ExternNodeRecord::numOutgoingPenCRs(int proc){
//
//  This function calculates and returns the number of penalty constraint
//  relations (lists of nodes, eqnID number lists, and DOF lists) that we
//  will be sending to processor proc.
//
    //what we're going to do is: look through the nodes in our local penNodes
    //table, and for each one owned by processor proc, add that constraint's
    //penID to a list of penIDs. The number of penIDs that get added to
    //the list is the number of outgoing penalty constraint relations for
    //processor proc.

    int i, j;
    int *tmpPenIDs = NULL;
    int lentmpIDs = 0;

    for(i=0; i<localNumPenCRs_; i++){
        for(j=0; j<lenLocPenNodes_[i]; j++){
            if (ownerProc(locPenNodes_[i][j]) == proc){
                if (!inList(tmpPenIDs,lentmpIDs, locPenID_[i])) {
                    appendIntList(&tmpPenIDs,&lentmpIDs,locPenID_[i]);
                }
            }
        }
    }

    delete [] tmpPenIDs;

    return(lentmpIDs);
}

//====================================================================
int* ExternNodeRecord::sendProcListPtr(int& lenSendProcs) {
    lenSendProcs = lenSendProcList_;
    return(sendProcList_);
}

//====================================================================
int* ExternNodeRecord::recvProcListPtr(int& lenRecvProcs) {
    lenRecvProcs = lenRecvProcList_;
    return(recvProcList_);
}

//====================================================================
GlobalID* ExternNodeRecord::externNodeList(int& lenExtNodeList){
//
//  This function forms a flat list of the nodeID's of all the external
//  nodes in this record, and returns a pointer to that list.
//
    int i;

    if (externNodeList_ != NULL) delete [] externNodeList_;
    externNodeList_ = NULL;
    externNodeList_ = new GlobalID[numAllExtNodes_];

    for(i=0; i<numAllExtNodes_; i++){
        externNodeList_[i] = allExtNodes_[i]->nodeID;
    }

    lenExtNodeList = numAllExtNodes_;
    return(externNodeList_);
}

//====================================================================
GlobalID** ExternNodeRecord::localNodeIDsPtr(int** lenLocNodeIDs) {
    *lenLocNodeIDs = lenSend_;
    return(localNodeIDs_);
}

//====================================================================
GlobalID** ExternNodeRecord::remoteNodeIDsPtr(int** lenRemoteNodeIDs) {
    *lenRemoteNodeIDs = lenRecv_;
    return(remoteNodeIDs_);
}

//====================================================================
bool ExternNodeRecord::isExternNode(GlobalID nodeID){
//
//  This is a query function. Returns true if node nodeID is an
//  external node.
//
    int i;

    for(i=0; i<numAllExtNodes_; i++) {
        if (allExtNodes_[i]->nodeID == nodeID) return(true);
    }

    return(false);
}
 
//====================================================================
int ExternNodeRecord::getNumMultCRs(GlobalID nodeID){
//
//  This function returns the total number of Lagrange CRs that remote 
//  node nodeID appears in.
//  Returns -1 if it isn't a remote node.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        return(allExtNodes_[index]->numMultCRs);
    }
    else {
        return(-1);
    }
}

//====================================================================
void ExternNodeRecord::setNumMultCRs(GlobalID nodeID, int numMultCRs){
//
//  This function sets the number of CRs that remote node nodeID
//  appears in.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        allExtNodes_[index]->numMultCRs = numMultCRs;
    }

    return;
}
 
 
//====================================================================
int ExternNodeRecord::getNumPenCRs(GlobalID nodeID){
//
//  This function returns the total number of penalty CRs that remote 
//  node nodeID appears in.
//  Returns -1 if it isn't a remote node.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        return(allExtNodes_[index]->numPenCRs);
    }
    else {
        return(-1);
    }
}

//====================================================================
void ExternNodeRecord::setNumPenCRs(GlobalID nodeID, int numPenCRs){
//
//  This function sets the number of CRs that remote node nodeID
//  appears in.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        allExtNodes_[index]->numPenCRs = numPenCRs;
    }

    return;
}
 
//====================================================================
int ExternNodeRecord::ownerProc(GlobalID nodeID){
//
//  This function returns the number of the processor that owns
//  external node nodeID.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        if (allExtNodes_[index]->ownerProc < 0) {
            cout << "ERROR, ExternNodeRecord::ownerProc, no owner for node "
                 << (int) nodeID << endl << flush;
            exit(0);
        }
        return(allExtNodes_[index]->ownerProc);
    }
    else {
        return(-1);
    }
}

//====================================================================
void ExternNodeRecord::ownerProc(GlobalID nodeID, int owner){
//
//  This function sets the ownerProc variable for remote
//  remote node nodeID.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        allExtNodes_[index]->ownerProc = owner;
    }
    else {
        cout << "ExternNodeRecord::ownerProc: ERROR, nodeID "
             << (int)nodeID << " not found!!!" << endl;
        exit(0);
    }

    return;
}
 
//====================================================================
int* ExternNodeRecord::procIDs(GlobalID nodeID, int& numprocs) {
//
//  This function returns a pointer to the list of processors that this
//  node is associated with.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >= 0) {
        numprocs = allExtNodes_[index]->numProcs;
        return(allExtNodes_[index]->procIDs);
    }
    else return(NULL);
}

//====================================================================
void ExternNodeRecord::procIDs(ExtNode *node, int* procs, int numprocs) {
//
//  This function sets the list of processors associated with this node.
//  If there are already processors associated with this node, the new 
//  ones are added to the list.
//
    int i, j;
    bool found;

    if (node->numProcs <= 0) {
        node->numProcs = numprocs;
        node->procIDs = new int[numprocs];
        for(i=0; i<numprocs; i++){
            node->procIDs[i] = procs[i];
        }
    }
    else {
        for(i=0; i<numprocs; i++) {
            found = false;
            for(j=0; j<node->numProcs; j++) {
                if (node->procIDs[j] == procs[i])
                    found = true;
            }
            if (!found) {
                //didn't find the proc number, so append it to
                //the existing array.

                appendIntList(&(node->procIDs), &node->numProcs, procs[i]);
            }
        }
    }

    return;
}

//====================================================================
int ExternNodeRecord::numSolnParams(GlobalID nodeID){
//
//  This function returns the number of solution parameters at
//  remote node nodeID.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        return(allExtNodes_[index]->numSolnParams);
    }
    else {
        return(-1);
    }
}

//====================================================================
void ExternNodeRecord::numSolnParams(GlobalID nodeID, int dof){
//
//  This function sets the number of solution parameters at
//  remote node nodeID.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0)
        allExtNodes_[index]->numSolnParams = dof;

    return;
}
 
//====================================================================
int ExternNodeRecord::globalEqn(GlobalID nodeID){
//
//  This function returns the global equation number for remote node
//  nodeID.
//  Returns -1 if nodeID is not in the allExtNodes_ list.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        return(allExtNodes_[index]->globalEqn);
    }
    else {
        return(-1);
    }
}

//====================================================================
void ExternNodeRecord::globalEqn(GlobalID nodeID, int eqnNum){
//
//  This function sets the global equation number of external
//  node nodeID.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0)
        allExtNodes_[index]->globalEqn = eqnNum;

    return;
}
 
//====================================================================
int* ExternNodeRecord::globalCREqn(GlobalID nodeID, int& numMultCRs){
//
//  This function returns the global equation number(s) of the Lagrange
//  constraint relation(s) that remote node nodeID appears in.
//  Returns NULL if nodeID is not in a Lagrange constraint found
//  in the allExtNodes_ list.
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        numMultCRs = allExtNodes_[index]->numMultCRs;
        return(allExtNodes_[index]->globalCREqn);
    }
    else {
        numMultCRs = -1;
        return(NULL);
    }
}

//====================================================================
void ExternNodeRecord::globalCREqn(GlobalID nodeID, int* eqnNum,
                                   int numMultCRs){
//
//  This function sets the global equation number(s) of the Lagrange
//  multiplier constraint relation(s) that external node nodeID appears
//  in.  We allow for the possibility that the node appears in more than 
//  one constraint relation. So for each incoming equation number, we look
//  through all of this node's equation numbers and add any that aren't 
//  already here.
//
    int i, j, index;
    bool found;

    if ((index = allExtNodeIndex(nodeID)) >=0) {
        for(i=0; i<numMultCRs; i++) {
            found = false;

//  we're only interested in constraints that generate new equations,
//  i.e., Lagrange Multiplier constraints.

            for(j=0; j<allExtNodes_[index]->numMultCRs; j++) {
                if (allExtNodes_[index]->globalCREqn[j] == eqnNum[i]){
                    found = true;
                }
            }
            if (!found) {
            
                //didn't find the equation number, so append it to
                //the existing array of equation numbers.

                appendIntList(&(allExtNodes_[index]->globalCREqn),
                             &(allExtNodes_[index]->numMultCRs), eqnNum[i]);
            }
        }
    }

    return;
}
 
//====================================================================
int ExternNodeRecord::allExtNodeIndex(GlobalID nodeID) {
//
//  This function returns an index into the array allExtNodes_
//  which can be used to access the one with global node ID nodeID.
//  Returns -1 if none match.
//  
    int i;

    for(i=0; i<numAllExtNodes_; i++){
        if (allExtNodes_[i]->nodeID == nodeID) return(i);
    }

    return(-1);
}
 
//====================================================================
int ExternNodeRecord::localNodeIndex(GlobalID nodeID) {
//
//  This function returns an index into the array localNodes_
//  which can be used to access the one with global node ID nodeID.
//  Returns -1 if none match.
//
    int i;

    for(i=0; i<numLocalNodes_; i++){
        if (localNodes_[i]->nodeID == nodeID) return(i);
    }

    return(-1);
}
 
//====================================================================
int ExternNodeRecord::remoteNodeIndex(GlobalID nodeID) {
//
//  This function returns an index into the array remoteNodes_
//  which can be used to access the one with global node ID nodeID.
//  Returns -1 if none match.
//
    int i;

    for(i=0; i<numRemoteNodes_; i++){
        if (remoteNodes_[i]->nodeID == nodeID) return(i);
    }

    return(-1);
}
 
//====================================================================
bool ExternNodeRecord::isPenaltyTerm(GlobalID nodeID) {
//
//  This function returns true if node nodeID is associated with a 
//  penalty constraint
//
    int index = allExtNodeIndex(nodeID);

    if (index <0) {
        cout << "ExternNodeRecord::isPenaltyTerm: ERROR node " << (int)nodeID
             << " not found!!!" << endl;
        exit(0);
    }
    
    return(allExtNodes_[index]->penaltyTerm);
}

//====================================================================
void ExternNodeRecord::penaltyTerm(GlobalID nodeID) {
//
//  This function sets to true the penaltyTerm field of the
//  appropriate node.
//

    int index = allExtNodeIndex(nodeID);

    if (index <0) {
        cout << "ExternNodeRecord::penaltyTerm: ERROR node " << (int)nodeID
             << " not found!!!" << endl;
        exit(0);
    }

    allExtNodes_[index]->penaltyTerm = true;
    allExtNodes_[index]->numPenCRs++;

    return;
}

//====================================================================
int ExternNodeRecord::getRecvProcIndex(int proc){
    int i;
    for(i=0; i<lenRecvProcList_; i++){
        if (recvProcList_[i] == proc) return(i);
    }

    return(-1);
}

//====================================================================
int ExternNodeRecord::getSendProcIndex(int proc){
    int i;
    for(i=0; i<lenSendProcList_; i++){
        if (sendProcList_[i] == proc) return(i);
    }

    return(-1);
}

//====================================================================
int ExternNodeRecord::getLocPenIDIndex(int penID) {
//
//  This function takes penID, and returns the index into the locPenID_ 
//  list such that locPenID_[index] == penID.
//
    int index;

    for(index=0; index<localNumPenCRs_; index++){
        if (locPenID_[index] == penID){
            return(index);
        }
    }

    return(-1);
}

//====================================================================
int ExternNodeRecord::getRemotePenIDIndex(int penID, int proc) {
//
//  This function takes penID and proc, and returns the index into the
//  remPenID_ list such that remPenID_[index] == penID and
//  remPenProcs_[index] == proc.
//
    int index;

    for(index=0; index<remoteNumPenCRs_; index++){
        if ((remPenID_[index] == penID) && (remPenProcs_[index] == proc)){
            return(index);
        }
    }

    return(-1);
}




//====================================================================
int ExternNodeRecord::getNumFields(GlobalID nodeID){
//
//  This function returns the number of fields found at an external
//  node with identifier nodeID
//
//  as always, returns -1 if this isn't an external node...
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >= 0) {
        return(allExtNodes_[index]->numFields);
    }
    else {
        return(-1);
    }
}


//====================================================================
void ExternNodeRecord::setNumFields(GlobalID nodeID, 
                                    int numFields){
//
//  This function sets the number of fields found at an external
//  node with identifier nodeID
//
//  here, if we don't find the given node, we just ignore this call
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >= 0) {
        allExtNodes_[index]->numFields = numFields;
    }
    return;
}
 
 
//====================================================================
int* ExternNodeRecord::getFieldIDList(GlobalID nodeID, 
                                      int& numFields) {
//
//  This function returns the list of field IDs corresponding to
//  the given nodeID
//
//  It returns NULL if the given nodeID is not found (i.e., that
//  node is not in the external node list)
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >= 0) {
        numFields = allExtNodes_[index]->numFields;
        return(allExtNodes_[index]->fieldIDs);
    }
    else {
        numFields = -1;
        return(NULL);
    }
}


//====================================================================
void ExternNodeRecord::setFieldIDList(GlobalID nodeID, 
                                      int* fieldIDList,
                                      int numFields) {

//  This function sets the list of field IDs corresponding to
//  the given nodeID

    int i, index;

//  attempt to exit gracefully on obvious error conditions

//$kdm   temp removal for debugging - uncomment and remove assert 
//$kdm   once this all works properly!
//$kdm
//$    if ((fieldIDList == NULL) || (numFields <= 0)) {
//$        return;

    assert (fieldIDList != NULL);  //$kdm - remove when tested
    assert (numFields > 0);        //$kdm - remove when tested

//  if the data seems ok, cache the passed information

    if ((index = allExtNodeIndex(nodeID)) >= 0) {
        allExtNodes_[index]->numFields = numFields;
        allExtNodes_[index]->fieldIDs = new int [numFields];
        for (i = 0; i < numFields; i++) {
            allExtNodes_[index]->fieldIDs[i] = fieldIDList[i];
//$            cout << "$kdm debug A : " << i << "  " << nodeID << "  "  
//$                 << numFields << "  " << fieldIDList[i] << endl;
        }
        return;
    }

//  couldn't find the node in the external node list, so we
//  might as well just ignore this call (if not, here's the place
//  for appropriate retributive error-handling!)

    else {
        cout << "there is some kinda TROUBLE in "
             << "ExternNodeRecord::setFieldIDList()..." << endl
             << "the nodeID with the problem is " << nodeID
             << endl << flush;
        return;
        
    }
}
 
 
//====================================================================
int* ExternNodeRecord::getFieldOffsetList(GlobalID nodeID, 
                                          int& numFields) {
//
//  This function returns the list of field offsets corresponding to
//  the given nodeID
//
//  It returns NULL if the given nodeID is not found (i.e., that
//  node is not in the external node list)
//
    int index;

    if ((index = allExtNodeIndex(nodeID)) >= 0) {
        numFields = allExtNodes_[index]->numFields;
        return(allExtNodes_[index]->fieldOffsets);
    }
    else {
        numFields = -1;
        return(NULL);
    }
}


//====================================================================
void ExternNodeRecord::setFieldOffsetList(GlobalID nodeID, 
                                          int* fieldOffsets,
                                          int numFields) {

//  This function sets the list of field offsets corresponding to
//  the given nodeID

    int i, index;

//  attempt to exit gracefully on obvious error conditions

//$kdm   temp removal for debugging - uncomment and remove assert 
//$kdm   once this all works properly!
//$kdm
//$    if ((fieldIDList == NULL) || (numFields <= 0)) {
//$        return;

    assert (fieldOffsets != NULL);  //$kdm - remove when tested
    assert (numFields > 0);         //$kdm - remove when tested

//  if the data seems ok, cache the passed information

    if ((index = allExtNodeIndex(nodeID)) >= 0) {
        allExtNodes_[index]->numFields = numFields;
        allExtNodes_[index]->fieldOffsets = new int [numFields];
        for (i = 0; i < numFields; i++) {
            allExtNodes_[index]->fieldOffsets[i] = fieldOffsets[i];
//$            cout << "$kdm debug B : " << i << "  " << nodeID << "  "  
//$                 << numFields << "  " << fieldOffsets[i] << endl;
        }
        return;
    }

//  couldn't find the node in the external node list, so we
//  might as well just ignore this call (if not, here's the place
//  for appropriate retributive error-handling!)

    else {
        cout << "there is some kinda TROUBLE in "
             << "ExternNodeRecord::setFieldOffsetList()..." << endl
             << "the nodeID with the problem is " << nodeID
             << endl << flush;
        return;
        
    }
}




//====================================================================
int ExternNodeRecord::getFieldOffset(GlobalID nodeID, int fieldID) {
//
//  This function returns the field offset for the external node
//  with identifier nodeID
//
//  as always, returns -1 if this isn't an external node...
//
//  this is arguably not the best place to put lookups for field
//  offsets in the external node container, because we end up 
//  searching on nodeID repeatedly.  This method's use in ISIS_SLE.cc
//  is thus an obvious candidate for some more direct access to the
//  elements within the external node container...

    int index;

    if ((index = allExtNodeIndex(nodeID)) >= 0) {

//  found the node, so let's try to find this field

        for (int i = 0; i < allExtNodes_[index]->numFields; i++) {
            if (fieldID == allExtNodes_[index]->fieldIDs[i]) {
                return(allExtNodes_[index]->fieldOffsets[i]);
            }
        }

//  couldn't find the field, so return an error

        return(-1);
    }

//  couldn't find the node, so also return an error

    else {
        return(-1);
    }
}
