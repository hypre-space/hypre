#include <iostream.h>
#include <assert.h>
#include <stdio.h>
#include "other/basicTypes.h"

#include "src/BCRecord.h"

//==========================================================================
BCRecord::BCRecord() {

    myFieldID = 0;
    myFieldSize = 0;
    myFieldOffset = 0;
    
    myAlpha = NULL;
    myBeta = NULL;
    myGamma = NULL;

    return;
}

//==========================================================================
BCRecord::~BCRecord() {

    if (myFieldSize > 0){
        delete [] myAlpha;
        delete [] myBeta;
        delete [] myGamma;
    }

    return;
}

//==========================================================================
int BCRecord::getFieldID() {
    return(myFieldID);
}

//==========================================================================
void BCRecord::setFieldID(int fieldID) {
    myFieldID = fieldID;
}

//==========================================================================
int BCRecord::getFieldSize() {
    return(myFieldSize);
}
//==========================================================================
void BCRecord::setFieldSize(int fieldSize) {
    myFieldSize = fieldSize;
}

//==========================================================================
int BCRecord::getFieldOffset() {
    return(myFieldOffset);
}
//==========================================================================
void BCRecord::setFieldOffset(int fieldOffset) {
    myFieldOffset = fieldOffset;
}
 
//==========================================================================
double *BCRecord::pointerToAlpha(int& fieldSize) {
    fieldSize = myFieldSize;
    return(myAlpha);
}
//==========================================================================
void BCRecord::allocateAlpha() {
    int i;
    assert (myFieldSize > 0);
    myAlpha = new double[myFieldSize];
    for (i = 0; i < myFieldSize; i++)
        myAlpha[i] = 0.0;
    return;
}
 
//==========================================================================
double *BCRecord::pointerToBeta(int& fieldSize) {
    fieldSize = myFieldSize;
    return(myBeta);
}
//==========================================================================
void BCRecord::allocateBeta() {
    int i;
    assert (myFieldSize > 0);
    myBeta = new double[myFieldSize];
    for (i = 0; i < myFieldSize; i++)
        myBeta[i] = 0.0;
    return;
}
 
//==========================================================================
double *BCRecord::pointerToGamma(int& fieldSize) {
    fieldSize = myFieldSize;
    return(myGamma);
}
//==========================================================================
void BCRecord::allocateGamma() {
    int i;
    assert (myFieldSize > 0);
    myGamma = new double[myFieldSize];
    for (i = 0; i < myFieldSize; i++)
        myGamma[i] = 0.0;
    return;
}

//==========================================================================
void BCRecord::dumpToScreen() {
    cout << "  myFieldID = " << myFieldID << endl;
    cout << "  myFieldSize = " << myFieldSize << endl;
    cout << "  myFieldOffset = " << myFieldOffset << endl;
    cout << "  the " << myFieldSize << " BC coeffs a, b, c :  " << endl;
	for (int i = 0; i < myFieldSize; ++i) {
        cout << "       " << myAlpha[i] 
             << "   " << myBeta[i] 
             << "   " << myGamma[i] << endl;
    }
    cout << endl;

    return;
}
