#ifndef __SharedNodeBuffer_h
#define __SharedNodeBuffer_h

// #include "NodePackets.h"

// The SharedNodeBuffer class is a container into which a processor
// can progressively store NodeControlPackets, lists of indices and
// lists of coefficients. SharedNodeBuffer internally aggregates
// this data into arrays that are grouped according to the destination
// processor. These arrays can then be sent in one big message (one
// message for each data type, that is. i.e., one big array of 
// NodeControlPackets, one big array of indices, etc., to each processor).

class SharedNodeBuffer {

  public:
    SharedNodeBuffer(int indicesSize, int coeffSize);
    ~SharedNodeBuffer();

    void addNodeControlPacket(NodeControlPacket &packet, int dest);

    void addIndices(int *indices, int numIndices, int dest);

    void addCoeffs(double *coeffs, int coeffLength, int dest);

    //query functions
    int numPacketDestProcs() const {return(numPacketDestProcs_);};
    int* packetDestProcsPtr(int &numProcs);
    int numIndexDestProcs() const {return(numIndexDestProcs_);};
    int* indexDestProcsPtr(int &numProcs);
    int numCoeffDestProcs() const {return(numCoeffDestProcs_);};
    int* coeffDestProcsPtr(int &numProcs);
    int numPacketUnits(int dest);
    int numIndexUnits(int dest);
    int numCoeffUnits(int dest);
    int indicesUnitLength() const {return(indicesSize_);};
    int coeffUnitLength() const {return(coeffSize_);};

    NodeControlPacket* packetPtr(int dest);
    int* indicesPtr(int dest);
    double* coeffPtr(int dest);

  private: //functions

    //function to return the index associated with a destination processor
    int packetDestProcIndex(int dest);
    int indicesDestProcIndex(int dest);
    int coeffDestProcIndex(int dest);

    //function to add a new destination processor.
    void addPacketDestProc(int dest);
    void addIndexDestProc(int dest);
    void addCoeffDestProc(int dest);

    void addRowToPacketTable();
    void addRowToIndicesTable();
    void addRowToCoeffsTable();

    void addPacketToRow(int row, NodeControlPacket &packet);
    void addIndicesToRow(int row, int *indices, int numIndices);
    void addCoeffsToRow(int row, double *coeffs, int coeffLength);

    void copyNodeControlPacket(NodeControlPacket *source,
                               NodeControlPacket *result);

    void copyIndicesUnit(int *source, int *result);

    void copyCoeffUnit(double *source, double *result);

  private: //data

    int numPacketDestProcs_;  //how many distinct destination processors
    int *packetDestProcs_;    //list of length numPacketDestProcs_.
    int numIndexDestProcs_;  //how many distinct destination processors
    int *indexDestProcs_;    //list of length numIndexDestProcs_.
    int numCoeffDestProcs_;  //how many distinct destination processors
    int *coeffDestProcs_;    //list of length numCoeffDestProcs_.

    int *numPacketUnits_; //numUnits_ is a list of length numDestProcs_.
    int *numIndexUnits_;  //its contents are the number of units (i.e., the
    int *numCoeffUnits_;  //number of packets, or the number of index-sets or
                          //the number of coefficient-sets) to be sent to
                          //each destination processor.

    int indicesSize_, coeffSize_; //unit-length of index-sets and
                                  //coefficient-sets, respectively.

    NodeControlPacket **packets_; //the packets_ table has one row for each
                                  //destination processor, and the length of
                                  //row i is numPacketUnits_[i].
    int **indices_;   //the indices_ table has one row for each destination
                      //processor, and the length of row i is
                      //numIndexUnits_[i]*indicesSize_.

    double **coeffs_; //the coeffs_ table has one row for each destination
                      //processor, and the length of row i is 
                      //numCoeffUnits_[i]*coeffSize_.
};

#endif

