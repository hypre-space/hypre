#include <stdlib.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "src/NodePackets.h"
#include "src/SharedNodeBuffer.h"

//----------------------------------------------------------------------
//----- Constructor ----------------------------------------------------
SharedNodeBuffer::SharedNodeBuffer(int indicesSize, int coeffSize){

    indicesSize_ = indicesSize;
    coeffSize_ = coeffSize;

    numPacketDestProcs_ = 0;
    numIndexDestProcs_ = 0;
    numCoeffDestProcs_ = 0;
    packetDestProcs_ = NULL;
    indexDestProcs_ = NULL;
    coeffDestProcs_ = NULL;

    numPacketUnits_ = NULL;
    numIndexUnits_ = NULL;
    numCoeffUnits_ = NULL;

    packets_ = NULL;
    indices_ = NULL;
    coeffs_ = NULL;

    return;
}

//----------------------------------------------------------------------
//----- Destructor -----------------------------------------------------
SharedNodeBuffer::~SharedNodeBuffer(){
    int i;

    for(i=0; i<numPacketDestProcs_; i++){
        delete [] packets_[i];
    }
    for(i=0; i<numIndexDestProcs_; i++){
        delete [] indices_[i];
    }
    for(i=0; i<numCoeffDestProcs_; i++){
        delete [] coeffs_[i];
    }

    if (numPacketDestProcs_>0){
        delete [] packets_;
        delete [] packetDestProcs_;
        delete [] numPacketUnits_;
    }
    if (numIndexDestProcs_>0){
        delete [] indices_;
        delete [] indexDestProcs_;
        delete [] numIndexUnits_;
    }
    if (numCoeffDestProcs_>0){
        delete [] coeffs_;
        delete [] coeffDestProcs_;
        delete [] numCoeffUnits_;
    }

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::addNodeControlPacket(NodeControlPacket &packet,
                                            int dest){
    int index = packetDestProcIndex(dest);

    if (index < 0) {
        //The incoming packet is the first for processor dest, so we'll
        //have to add 'dest' to the processor list, lengthen the
        //numPacketUnits_ list, and add a new row to the packets_ table.

        //That's all done by this function
        addPacketDestProc(dest);

        index = packetDestProcIndex(dest);
    }

    //add the packet to row 'index'
    addPacketToRow(index, packet);

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::addIndices(int *indices, int numIndices,
                                  int dest){
    int index = indicesDestProcIndex(dest);

    if (index < 0) {
        //The incoming indices are the first for processor dest, so we'll
        //have to add 'dest' to the processor list, lengthen the
        //numIndexUnits_ list, and add a new row to the indices_ table.

        //That's all done by this function
        addIndexDestProc(dest);

        index = indicesDestProcIndex(dest);
    }

    //add the indices to row 'index'
    addIndicesToRow(index, indices, numIndices);

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::addCoeffs(double *coeffs, int coeffLength,
                                 int dest){
    int index = coeffDestProcIndex(dest);

    if (index < 0) {
        //The incoming coeffs are the first for processor dest, so we'll
        //have to add 'dest' to the processor list, lengthen the
        //numCoeffUnits_ lists, and add a new row to the coeffs_ table.

        //That's all done by this function
        addCoeffDestProc(dest);

        index = coeffDestProcIndex(dest);
    }

    //add the coeffs to row 'index'
    addCoeffsToRow(index, coeffs, coeffLength);

    return;
}

//----------------------------------------------------------------------
int* SharedNodeBuffer::packetDestProcsPtr(int &numProcs){
    numProcs = numPacketDestProcs_;
    return(packetDestProcs_);
}

//----------------------------------------------------------------------
int* SharedNodeBuffer::indexDestProcsPtr(int &numProcs){
    numProcs = numIndexDestProcs_;
    return(indexDestProcs_);
}

//----------------------------------------------------------------------
int* SharedNodeBuffer::coeffDestProcsPtr(int &numProcs){
    numProcs = numCoeffDestProcs_;
    return(coeffDestProcs_);
}

//----------------------------------------------------------------------
int SharedNodeBuffer::numPacketUnits(int dest) {
    int index = packetDestProcIndex(dest);

    if (index < 0) {
        cout << "ERROR in SharedNodeBuffer::numPacketUnits, no dest proc "
             << dest << endl << flush;
        return(-1);
    }
    else {
        return(numPacketUnits_[index]);
    }
}

//----------------------------------------------------------------------
int SharedNodeBuffer::numIndexUnits(int dest) {
    int index = indicesDestProcIndex(dest);

    if (index < 0) {
        cout << "ERROR in SharedNodeBuffer::numIndexUnits, no dest proc "
             << dest << endl << flush;
        return(-1);
    }
    else {
        return(numIndexUnits_[index]);
    }
}

//----------------------------------------------------------------------
int SharedNodeBuffer::numCoeffUnits(int dest) {
    int index = coeffDestProcIndex(dest);

    if (index < 0) {
        cout << "ERROR in SharedNodeBuffer::numCoeffUnits, no dest proc "
             << dest << endl << flush;
        return(-1);
    }
    else {
        return(numCoeffUnits_[index]);
    }
}
 
//----------------------------------------------------------------------
NodeControlPacket* SharedNodeBuffer::packetPtr(int dest){
    int index = packetDestProcIndex(dest);

    if (index < 0) {
        cout << "ERROR in SharedNodeBuffer::packetPtr, no dest proc "
             << dest << endl << flush;
        return(NULL);
    }
    else {
        return(packets_[index]);
    }
}

//----------------------------------------------------------------------
int* SharedNodeBuffer::indicesPtr(int dest){
    int index = indicesDestProcIndex(dest);

    if (index < 0) {
        cout << "ERROR in SharedNodeBuffer::indicesPtr, no dest proc "
             << dest << endl << flush;
        return(NULL);
    }
    else {
        return(indices_[index]);
    }
}

//----------------------------------------------------------------------
double* SharedNodeBuffer::coeffPtr(int dest){
    int index = coeffDestProcIndex(dest);

    if (index < 0) {
        cout << "ERROR in SharedNodeBuffer::coeffPtr, no dest proc "
             << dest << endl << flush;
        return(NULL);
    }
    else {
        return(coeffs_[index]);
    }
}

//----------------------------------------------------------------------
int SharedNodeBuffer::packetDestProcIndex(int dest){
//
// Return the index associated with processor 'dest'. 
// Return -1 if 'dest' isn't in the packetDestProcs_ list.
//

    for(int i=0; i<numPacketDestProcs_; i++){
        if (packetDestProcs_[i] == dest) return(i);
    }

    return(-1);
}

//----------------------------------------------------------------------
int SharedNodeBuffer::indicesDestProcIndex(int dest){
//
// Return the index associated with processor 'dest'.
// Return -1 if 'dest' isn't in the indexDestProcs_ list.
//

    for(int i=0; i<numIndexDestProcs_; i++){
        if (indexDestProcs_[i] == dest) return(i);
    }

    return(-1);
}
 
//----------------------------------------------------------------------
int SharedNodeBuffer::coeffDestProcIndex(int dest){
//
// Return the index associated with processor 'dest'.
// Return -1 if 'dest' isn't in the coeffDestProcs_ list.
//

    for(int i=0; i<numCoeffDestProcs_; i++){
        if (coeffDestProcs_[i] == dest) return(i);
    }

    return(-1);
}
 
//----------------------------------------------------------------------
void SharedNodeBuffer::addPacketDestProc(int dest) {
//
// Add processor 'dest' to the packetDestProcs_ list, lengthen the
// numPacketUnits_ list, and add a new row to the packets_ table.
//
    int i;

    //first, lengthen the packetDestProcs_ list.
    int *newProcList = new int[numPacketDestProcs_+1];
    for(i=0; i<numPacketDestProcs_; i++) newProcList[i] = packetDestProcs_[i];
    newProcList[numPacketDestProcs_] = dest;

    delete [] packetDestProcs_;
    packetDestProcs_ = newProcList;

    //now lengthen the numPacketUnits_ list.
    int *newUnitsList = new int[numPacketDestProcs_+1];
    for(i=0; i<numPacketDestProcs_; i++) newUnitsList[i] = numPacketUnits_[i];
    newUnitsList[numPacketDestProcs_] = 0;

    delete [] numPacketUnits_;
    numPacketUnits_ = newUnitsList;

    //finally, add a row to the table.
    addRowToPacketTable();

    numPacketDestProcs_++;

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::addIndexDestProc(int dest) {
//
// Add processor 'dest' to the indexDestProcs_ list, lengthen the
// numIndexUnits_ list, and add a new row to the indices_ table.
//
    int i;

    //first, lengthen the indexDestProcs_ list.
    int *newProcList = new int[numIndexDestProcs_+1];
    for(i=0; i<numIndexDestProcs_; i++) newProcList[i] = indexDestProcs_[i];
    newProcList[numIndexDestProcs_] = dest;

    delete [] indexDestProcs_;
    indexDestProcs_ = newProcList;

    //now lengthen the numIndexUnits_ list.
    int *newUnitsList = new int[numIndexDestProcs_+1];
    for(i=0; i<numIndexDestProcs_; i++) newUnitsList[i] = numIndexUnits_[i];
    newUnitsList[numIndexDestProcs_] = 0;

    delete [] numIndexUnits_;
    numIndexUnits_ = newUnitsList;

    //finally, add a row to the table.
    addRowToIndicesTable();

    numIndexDestProcs_++;

    return;
}
 
//----------------------------------------------------------------------
void SharedNodeBuffer::addCoeffDestProc(int dest) {
//
// Add processor 'dest' to the coeffDestProcs_ list, lengthen the
// numCoeffUnits_ list, and add a new row to the coeffs_ table.
//
    int i;

    //first, lengthen the coeffDestProcs_ list.
    int *newProcList = new int[numCoeffDestProcs_+1];
    for(i=0; i<numCoeffDestProcs_; i++) newProcList[i] = coeffDestProcs_[i];
    newProcList[numCoeffDestProcs_] = dest;

    delete [] coeffDestProcs_;
    coeffDestProcs_ = newProcList;

    //now lengthen the numCoeffUnits_ list.
    int *newUnitsList = new int[numCoeffDestProcs_+1];
    for(i=0; i<numCoeffDestProcs_; i++) newUnitsList[i] = numCoeffUnits_[i];
    newUnitsList[numCoeffDestProcs_] = 0;

    delete [] numCoeffUnits_;
    numCoeffUnits_ = newUnitsList;

    //finally, add a row to the table.
    addRowToCoeffsTable();

    numCoeffDestProcs_++;

    return;
}
 
//----------------------------------------------------------------------
void SharedNodeBuffer::addRowToPacketTable(){
    int i, j;

    NodeControlPacket **newTable = new NodeControlPacket*[numPacketDestProcs_+1];

    for(i=0; i<numPacketDestProcs_; i++){
        newTable[i] = new NodeControlPacket[numPacketUnits_[i]];
        for(j=0; j<numPacketUnits_[i]; j++){
            copyNodeControlPacket(&packets_[i][j], &newTable[i][j]);
        }
    }
    newTable[numPacketDestProcs_] = NULL;

    for(i=0; i<numPacketDestProcs_; i++){
        delete [] packets_[i];
    }
    delete [] packets_;

    packets_ = newTable;

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::addRowToIndicesTable(){
    int i, j;

    int **newTable = new int*[numIndexDestProcs_+1];

    for(i=0; i<numIndexDestProcs_; i++){
        newTable[i] = new int[numIndexUnits_[i]*indicesSize_];
        for(j=0; j<numIndexUnits_[i]; j++){
            copyIndicesUnit(&(indices_[i][j*indicesSize_]),
                            &(newTable[i][j*indicesSize_]));
        }
    }
    newTable[numIndexDestProcs_] = NULL;

    for(i=0; i<numIndexDestProcs_; i++){
        delete [] indices_[i];
    }
    delete [] indices_;

    indices_ = newTable;

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::addRowToCoeffsTable(){
    int i, j;

    double **newTable = new double*[numCoeffDestProcs_+1];

    for(i=0; i<numCoeffDestProcs_; i++){
        newTable[i] = new double[numCoeffUnits_[i]*coeffSize_];
        for(j=0; j<numCoeffUnits_[i]; j++){
            copyCoeffUnit(&(coeffs_[i][j*coeffSize_]),
                            &(newTable[i][j*coeffSize_]));
        }
    }
    newTable[numCoeffDestProcs_] = NULL;

    for(i=0; i<numCoeffDestProcs_; i++){
        delete [] coeffs_[i];
    }
    delete [] coeffs_;

    coeffs_ = newTable;

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::copyNodeControlPacket(NodeControlPacket *source,
                                             NodeControlPacket *result){
    result->nodeID = source->nodeID;
    result->numEqns = source->numEqns;
    result->eqnNumber = source->eqnNumber;
    result->numElems = source->numElems;
    result->numIndices = source->numIndices;
    result->numPenCRs = source->numPenCRs;
    result->numMultCRs = source->numMultCRs;

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::addPacketToRow(int row, NodeControlPacket &packet){

    NodeControlPacket *newRow = new NodeControlPacket[numPacketUnits_[row]+1];

    for(int i=0; i<numPacketUnits_[row]; i++){
        copyNodeControlPacket(&(packets_[row][i]), &(newRow[i]));
    }
    copyNodeControlPacket(&packet, &(newRow[numPacketUnits_[row]]));

    delete [] packets_[row];
    packets_[row] = newRow;

    numPacketUnits_[row]++;

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::addIndicesToRow(int row, int *indices, int numIndices){

    int i;
    int *newRow = new int[indicesSize_*(numIndexUnits_[row]+1)];

    for(i=0; i<numIndexUnits_[row]; i++){
        copyIndicesUnit(&(indices_[row][i*indicesSize_]),
                        &(newRow[i*indicesSize_]));
    }

    int offset = indicesSize_*numIndexUnits_[row];
    for(i=0; i<numIndices; i++){
        newRow[offset+i] = indices[i];
    }

    delete [] indices_[row];
    indices_[row] = newRow;

    numIndexUnits_[row]++;

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::addCoeffsToRow(int row, double *coeffs, int coeffLength){

    int i;
    double *newRow = new double[coeffSize_*(numCoeffUnits_[row]+1)];

    for(i=0; i<numCoeffUnits_[row]; i++){
        copyCoeffUnit(&(coeffs_[row][i*coeffSize_]),
                        &(newRow[i*coeffSize_]));
    }

    int offset = coeffSize_*numCoeffUnits_[row];
    for(i=0; i<coeffLength; i++){
        newRow[offset+i] = coeffs[i];
    }

    delete [] coeffs_[row];
    coeffs_[row] = newRow;

    numCoeffUnits_[row]++;

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::copyIndicesUnit(int *source, int *result){

    for(int i=0; i<indicesSize_; i++) result[i] = source[i];

    return;
}

//----------------------------------------------------------------------
void SharedNodeBuffer::copyCoeffUnit(double *source, double *result){

    for(int i=0; i<coeffSize_; i++) result[i] = source[i];

    return;
}

