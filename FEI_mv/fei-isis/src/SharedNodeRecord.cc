#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>

#include "other/basicTypes.h"
#include "src/SharedNodeRecord.h"

/*==================================================================*/
SharedNodeRecord::SharedNodeRecord() {
/*
   Constructor
*/
    sharedNodes_ = NULL;
    numSharedNodes_ = 0;
    localProc_ = -1;
    initCompleted_ = false;

    localShared_ = NULL;
    numLocalShared_ = 0;

    remoteShared_ = NULL;
    numRemoteShared_ = 0;

    numSharingProcs_ = 0;
    sharingProcs_ = NULL;

    return;
}

/*==================================================================*/
SharedNodeRecord::~SharedNodeRecord() {
/*
   Destructor
*/
    for(int i=0; i<numSharedNodes_; i++){
        if (sharedNodes_[i].numProcs > 0){
            delete [] sharedNodes_[i].procIDs;
            delete [] sharedNodes_[i].remoteElems;
        }
    }
    delete [] sharedNodes_;

    delete [] localShared_;
    delete [] remoteShared_;

    if (numSharingProcs_ > 0)
        delete [] sharingProcs_;

    return;
}

/*==================================================================*/
void SharedNodeRecord::sharedNodes(const GlobalID *nodeIDs,
         int numNodes, const int *const *procs,
         const int *lenProcs, int localProc) {
/*
   This function is used for passing in the parameters that we get from
   the FEI function initSharedNodeSet.

   This is where the array of shared nodes will be established or added
   to.
*/
 
    int i, j, jj, start;
    SharedNode* newPtr = NULL;

    if (localProc_ < 0) {
        localProc_ = localProc;
    }
    if (localProc_ != localProc) {
        printf("SharedNodeRecord::sharedNodes: localProc parameter doesn't\n");
        printf("match previously recorded local proc number!!\n");
        printf("parameter localProc: %d, internal localProc_: %d\n",
               localProc, localProc_);
    }

    start = 0;

    if (numNodes <= 0) return; //zero length shared node list, do nothing.

    if (numSharedNodes_ <= 0) {
        //We need to establish the array of shared nodes. i.e., these
        //nodes are the first ones we've been given, so start the array
        //of shared nodes and put the first node in now.

        sharedNodes_ = new SharedNode[1];
        numSharedNodes_ = 1;

        sharedNodes_[0].nodeID = nodeIDs[0];
        sharedNodes_[0].numProcs = 0;
        sharedNodes_[0].procIDs = NULL;
        procIDs(nodeIDs[0], procs[0], lenProcs[0]);
        sharedNodes_[0].ownerProc = -1;
        sharedNodes_[0].equationNumber = -1;
        sharedNodes_[0].numEquations = 0;
        sharedNodes_[0].localElems = 0;
        sharedNodes_[0].equationLengths = NULL;
        sharedNodes_[0].remoteElems = NULL;

        start = 1;
    }

    //now loop through any other nodes in the input list
    for(i=start; i<numNodes; i++) {
        if (sharedNodeIndex(nodeIDs[i]) >= 0) {
            //This node is already here, so just add these procs to its
            //record.
            procIDs(nodeIDs[i], procs[i], lenProcs[i]);
        }
        else {
            //This is a new node, so we'll have to lengthen the array
            //of shared nodes and add it in.

            newPtr = new SharedNode[numSharedNodes_+1];

            //copy the old data into the new array
            for(j=0; j<numSharedNodes_; j++) {
                newPtr[j].nodeID = sharedNodes_[j].nodeID;
                newPtr[j].numProcs = sharedNodes_[j].numProcs;
                newPtr[j].procIDs = NULL;
                newPtr[j].ownerProc = sharedNodes_[j].ownerProc;
                newPtr[j].equationNumber = sharedNodes_[j].equationNumber;
                newPtr[j].numEquations = sharedNodes_[j].numEquations;
                newPtr[j].localElems = sharedNodes_[j].localElems;
                newPtr[j].equationLengths = NULL;
                newPtr[j].remoteElems = NULL;

                newPtr[j].procIDs = new int[sharedNodes_[j].numProcs];
                for(jj=0; jj<sharedNodes_[j].numProcs; jj++) {
                    newPtr[j].procIDs[jj] = sharedNodes_[j].procIDs[jj];
                }
            }

            //now put in the new node.
            newPtr[numSharedNodes_].nodeID = nodeIDs[i];
            newPtr[numSharedNodes_].numProcs = 0;
            newPtr[numSharedNodes_].procIDs = NULL;
            newPtr[numSharedNodes_].ownerProc = -1;
            newPtr[numSharedNodes_].equationNumber = -1;
            newPtr[numSharedNodes_].numEquations = 0;
            newPtr[numSharedNodes_].localElems = 0;
            newPtr[numSharedNodes_].equationLengths = NULL;
            newPtr[numSharedNodes_].remoteElems = NULL;

            //now tidy up.
            for(j=0; j<numSharedNodes_; j++){
                delete [] sharedNodes_[j].procIDs;
            }
            delete [] sharedNodes_;
            sharedNodes_ = newPtr;
            numSharedNodes_++;

            //finally, put in the procs associated with this node
            procIDs(nodeIDs[i], procs[i], lenProcs[i]);

        }
    }

    return;
} 
 
/*==================================================================*/
void SharedNodeRecord::initComplete() {
/*
   This function should be called after all shared nodes have been
   loaded in (through the sharedNodes(...) function).
   This function will set the ownerProc member variable for each of the
   shared nodes.
   This function also builds the lists of local-shared and
   remote-shared nodes.
   This function also allocates the remoteElems list for each node.
*/
    int i, j;
    bool found;

    for(i=0; i<numSharedNodes_; i++){
        //first, check to see if the local processor is included in this
        //shared node's list of sharing processors.
        found = false;
        for(j=0; j<sharedNodes_[i].numProcs; j++){
            if (sharedNodes_[i].procIDs[j] == localProc_) {
                found = true;
            }
        }

        //if the local processor wasn't included in the list of sharing
        //processors, put it in.
        if (!found) appendIntList(&(sharedNodes_[i].procIDs),
                                  &(sharedNodes_[i].numProcs), localProc_);

        //now set the ownerProc variable for this shared node.
        sharedNodes_[i].ownerProc = chooseOwner(sharedNodes_[i].procIDs,
                                                sharedNodes_[i].numProcs);

        //now add this node to the local-shared list if we are the owner,
        //or the remote-shared list otherwise.
        if (sharedNodes_[i].ownerProc == localProc_) {
            appendGlobalIDList(&localShared_, &numLocalShared_,
                               sharedNodes_[i].nodeID);
        }
        else {
            appendGlobalIDList(&remoteShared_, &numRemoteShared_,
                               sharedNodes_[i].nodeID);
        }

        //now allocate the remoteElems list for this node.
        sharedNodes_[i].remoteElems = new int[sharedNodes_[i].numProcs];
        for(j=0; j<sharedNodes_[i].numProcs; j++){
            sharedNodes_[i].remoteElems[j] = 0;
        }
    }

    buildSharingProcList();

    initCompleted_ = true;

    return;
}

/*==================================================================*/
void SharedNodeRecord::buildSharingProcList() {
//
//This function builds the list of processors that we'll be recv'ing
//indices from. i.e., processors that share nodes that we own.
//

    for(int i=0; i<numSharedNodes_; i++) {
        if (sharedNodes_[i].ownerProc == localProc_) {
            for(int j=0; j<sharedNodes_[i].numProcs; j++) {
                int proc = sharedNodes_[i].procIDs[j];
                bool alreadyInList = false;

                if (proc != localProc_) {
                    for(int k=0; k<numSharingProcs_; k++) {
                        if (proc == sharingProcs_[k])
                            alreadyInList = true;
                    }
                    if (!alreadyInList) {
                        appendIntList(&sharingProcs_, &numSharingProcs_,
                                      proc);
                    }
                }
            }
        }
    }
}

/*==================================================================*/
int* SharedNodeRecord::pointerToSharingProcs(int& numProcs) {
    numProcs = numSharingProcs_;
    return(sharingProcs_);
}

/*==================================================================*/
void SharedNodeRecord::printem() {

    int i, j;

    for(i=0; i<numSharedNodes_; i++){
        printf("%d: nodeID %d, localElems %d, numProcs %d, ownerProc %d\n",
               localProc_,
               (int)sharedNodes_[i].nodeID, sharedNodes_[i].localElems,
               sharedNodes_[i].numProcs, sharedNodes_[i].ownerProc);
        printf("%d: ---- ",localProc_);
        for(j=0; j<sharedNodes_[i].numProcs; j++) {
            printf("%d ",sharedNodes_[i].procIDs[j]);
        }
        printf("\n\n");
    }
    return;
}

/*==================================================================*/
GlobalID *SharedNodeRecord::pointerToLocalShared(int& numLocalShared) {
    if (numLocalShared_ > 0) {
        numLocalShared = numLocalShared_;
        return(localShared_);
    }
    else {
        numLocalShared = 0;
        return(NULL);
    }
}

/*==================================================================*/
GlobalID *SharedNodeRecord::pointerToRemoteShared(int& numRemoteShared) {
    if (numRemoteShared_ > 0) {
        numRemoteShared = numRemoteShared_;
        return(remoteShared_);
    }
    else {
        numRemoteShared = 0;
        return(NULL);
    }
}
 
/*==================================================================*/
int SharedNodeRecord::ownerProc(GlobalID nodeID) {

    int index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        return(sharedNodes_[index].ownerProc);
    }
    else {
        return(-1);
    }
}

/*==================================================================*/
int SharedNodeRecord::numMesgsLocalNodes() {

    int i, mesgs;

    if (!initCompleted_){
        printf("SharedNodeRecord::numMesgsLocalNodes:\n");
        printf("             you must call SharedNodeRecord::initComplete\n");
        printf("             before you call this function.\n");
        exit(0);
    }

    mesgs = 0;
    for(i=0; i<numSharedNodes_; i++) {
        if (localProc_ == sharedNodes_[i].ownerProc) 
            mesgs += (sharedNodes_[i].numProcs - 1);
    }

    return(mesgs);
}

/*==================================================================*/
int SharedNodeRecord::numMesgsRemoteNodes() {

    int i, mesgs;

    if (!initCompleted_){
        printf("SharedNodeRecord::numMesgsRemoteNodes:\n");
        printf("             you must call SharedNodeRecord::initComplete\n");
        printf("             before you call this function.\n");
        exit(0);
    }

    mesgs = 0;
    for(i=0; i<numSharedNodes_; i++) {
        if (localProc_ != sharedNodes_[i].ownerProc)
            mesgs++;
    }

    return(mesgs);
}
 
/*==================================================================*/
int SharedNodeRecord::chooseOwner(int *list, int lenList) {
/*
   This function takes a list (of processor IDs) and returns the
   number of the owning processor, which is designated to be the lowest
   numbered processor.
*/
    int i, owner;

    owner = list[0];
    for(i=1; i<lenList; i++){
        if (owner > list[i]) owner = list[i];
    }

    return(owner);
}

/*==================================================================*/
int *SharedNodeRecord::pointerToProcIDs(GlobalID nodeID, int& numProcs) {

    int index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        numProcs = sharedNodes_[index].numProcs;
        return(sharedNodes_[index].procIDs);
    }
    else {
        numProcs = -1;
        return(NULL);
    }
}

/*==================================================================*/
int SharedNodeRecord::numEquations(GlobalID nodeID) {
/*
   This function returns the number of equations that are associated
   with node nodeID.
*/
    int index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        return(sharedNodes_[index].numEquations);
    }
    else {
        cout << "int SharedNodeRecord::numEquations: ERROR, proc "
             << localProc_ << " doesn't have nodeID " << nodeID << endl;
        return(-1);
    }
}

/*==================================================================*/
void SharedNodeRecord::numEquations(GlobalID nodeID, int numEqns) {
/*
   This function sets the number of equations that processor procID
   thinks this node has.
*/
    int index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        sharedNodes_[index].numEquations = numEqns;
    }
    else {
        cout << "SharedNodeRecord::numEquations: ERROR, proc " << localProc_
             << " doesn't have nodeID " << (int)nodeID << endl;
        abort();
    }
    return;
}

/*==================================================================*/
int SharedNodeRecord::remoteElems(GlobalID nodeID, int procID) const {
/*
   This function returns the number of elements this node appears in
   on processor procID.
*/
    int i, index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        for(i=0; i<sharedNodes_[index].numProcs; i++){
            if (sharedNodes_[index].procIDs[i] == procID){
                return(sharedNodes_[index].remoteElems[i]);
            }
        }
        cout << "ERROR, int SharedNodeRecord::remoteElems: no procID " 
            << procID <<endl;
        abort();
    }
    else {
        cout << "ERROR, int SharedNodeRecord::remoteElems: proc " 
             << localProc_ << ", no nodeID " << (int)nodeID <<endl;
        abort();
    }

    return(-1);
}

/*==================================================================*/
void SharedNodeRecord::remoteElems(GlobalID nodeID, int procID, int numRElems) {
/*
   This function stores the number of elements this node appears in
   on processor procID.
*/
    int i, index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        for(i=0; i<sharedNodes_[index].numProcs; i++){
            if (sharedNodes_[index].procIDs[i] == procID){
                sharedNodes_[index].remoteElems[i] = numRElems;
            }
        }
    }
    else {
        cout << "ERROR, void SharedNodeRecord::remoteElems: proc "
            << localProc_ << ", no nodeID " << (int)nodeID <<endl;
        abort();
    }

    return;
}

/*==================================================================*/
int SharedNodeRecord::totalRemoteElems() {
/*
   This function returns the total number of elements on other processors
   that contain shared nodes which are owned by this processor.
*/
    int i, j, total;

    total = 0;
    for(i=0; i<numSharedNodes_; i++) {
        if (sharedNodes_[i].ownerProc == localProc_) {
            for(j=0; j<sharedNodes_[i].numProcs; j++) {
                total += sharedNodes_[i].remoteElems[j];
            }
        }
    }

    return(total);
}

/*==================================================================*/
int SharedNodeRecord::equationNumber(GlobalID nodeID) {
    int index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        return(sharedNodes_[index].equationNumber);
    }
    else {
        cout << "ERROR, int SharedNodeRecord::equationNumber: "
             << "nodeID " << (int)nodeID << " not found on proc "
             << localProc_ << "." << endl;
        return(-1);
    }
}

/*==================================================================*/
void SharedNodeRecord::equationNumber(GlobalID nodeID, int eqnNum) {
    int index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        sharedNodes_[index].equationNumber = eqnNum;
    }
    else {
        cout << "ERROR, void SharedNodeRecord::equationNumber: "
             << "nodeID " << (int)nodeID << " not found on proc "
             << localProc_ << "." << endl;
    }
    return;
}

/*==================================================================*/
int SharedNodeRecord::sharedNodeIndex(GlobalID nodeID) const {
/*
   The shared nodes are stored in an array called sharedNodes_.
   This function returns an index into that array
   which can be used to access the one with global node ID nodeID.
   Returns -1 if none match.
*/
    int i;

    for(i=0; i<numSharedNodes_; i++){
        if (sharedNodes_[i].nodeID == nodeID) return(i);
    }

    return(-1);
}
 
/**====================================================================**/
int SharedNodeRecord::isShared(GlobalID nodeID){
/*
   This is a query function. Returns the number of the owning processor
   if node nodeID is a shared node.
   Returns -1 otherwise.
*/
    int i;

    for(i=0; i<numSharedNodes_; i++) {
        if (sharedNodes_[i].nodeID == nodeID) 
            return(sharedNodes_[i].ownerProc);
    }

    return(-1);
}
 
/*==================================================================*/
void SharedNodeRecord::appendIntList(int** list, int* lenList, int newItem){
/*
   This function appends an integer to a list of integers.
   Yeah, yeah, I know, this should be a template.
*/
    int i;

    //first we allocate the new list
    int* newList = new int[*lenList+1];

    //now we copy the old stuff into the new list
    for(i=0; i<(*lenList); i++) newList[i] = (*list)[i];

    //now put in the new item
    newList[*lenList] = newItem;

    //and finally delete the old memory and set the pointer to
    //point to the new memory
    if (*lenList > 0) delete [] (*list);
    *list = newList;
    (*lenList) += 1;

    return;
}
 
/**====================================================================**/
void SharedNodeRecord::appendGlobalIDList(GlobalID** list, int* lenList,
                                          GlobalID newItem){
/*
   This function appends a GlobalID to a list of GlobalIDs.
   Yeah, yeah, I know, this should be a template.
*/
    int i;

    //first we allocate the new list
    GlobalID* newList = new GlobalID[*lenList+1];

    //now we copy the old stuff into the new list
    for(i=0; i<(*lenList); i++) newList[i] = (*list)[i];

    //now put in the new item
    newList[*lenList] = newItem;

    //and finally delete the old memory and set the pointer to
    //point to the new memory
    if (*lenList > 0) delete [] (*list);
    *list = newList;
    (*lenList) += 1;

    return;
}
 
/**====================================================================**/
int* SharedNodeRecord::procIDs(GlobalID nodeID, int& numprocs) {
/*
   This function returns a pointer to the list of processors that this
   node is associated with.
*/

    int index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        numprocs = sharedNodes_[index].numProcs;
        return(sharedNodes_[index].procIDs);
    }
    else return(NULL);
}
 
/**====================================================================**/
int SharedNodeRecord::localElems(GlobalID nodeID) const {
/*
   This function returns the number of local elements that node nodeID
   appears in.
*/

    int index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        return(sharedNodes_[index].localElems);
    }
    else return(-1);
}
 
/**====================================================================**/
void SharedNodeRecord::localElems(GlobalID nodeID, int numElems){
/*
   This function sets the number of local elements that node nodeID
   appears in.
*/

    int index;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        sharedNodes_[index].localElems = numElems;
    }
}
 
/**====================================================================**/
void SharedNodeRecord::procIDs(GlobalID nodeID, const int* procs,
                               int numprocs) {
/*
   This function sets the list of processors associated with this node.
   If there are already processors associated with this node, the new ones
   are added to the list.
*/
    int i, j, index;
    bool found;

    if ((index = sharedNodeIndex(nodeID)) >= 0) {
        if (sharedNodes_[index].numProcs <= 0) {
            sharedNodes_[index].numProcs = numprocs;
            sharedNodes_[index].procIDs = new int[numprocs];
            for(i=0; i<numprocs; i++){
                sharedNodes_[index].procIDs[i] = procs[i];
            }
        }
        else {
            for(i=0; i<numprocs; i++) {
                found = false;
                for(j=0; j<sharedNodes_[index].numProcs; j++) {
                    if (sharedNodes_[index].procIDs[j] == procs[i])
                        found = true;
                }
                if (!found) {
                    //didn't find the proc number, so append it to
                    //the existing array.

                    int tmp = sharedNodes_[index].numProcs;

                    appendIntList(&(sharedNodes_[index].procIDs),
                                 &tmp, procs[i]);
                    sharedNodes_[index].numProcs = tmp;
                }
            }
        }
    }

    return;
}
 
