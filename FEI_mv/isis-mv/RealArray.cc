/*
          This file is part of Sandia National Laboratories
          copyrighted software.  You are legally liable for any
          unauthorized use of this software.

          NOTICE:  The United States Government has granted for
          itself and others acting on its behalf a paid-up,
          nonexclusive, irrevocable worldwide license in this
          data to reproduce, prepare derivative works, and
          perform publicly and display publicly.  Beginning five
          (5) years after June 5, 1997, the United States
          Government is granted for itself and others acting on
          its behalf a paid-up, nonexclusive, irrevocable
          worldwide license in this data to reproduce, prepare
          derivative works, distribute copies to the public,
          perform publicly and display publicly, and to permit
          others to do so.

          NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED
          STATES DEPARTMENT OF ENERGY, NOR SANDIA CORPORATION,
          NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS
          OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
          RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
          USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT, OR
          PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
          INFRINGE PRIVATELY OWNED RIGHTS.
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream.h>
#include "RealArray.h"

/*====================================================================*/
RealArray::RealArray(int n) {
/*
    Constructs array of size n, initializes its contents to 0.
*/

    if (n == 0) {
        array_ = NULL;
        size_ = 0;
    }
    else {
        array_ = new double[n];
        size_ = n;

        for(int i=0; i<size_; i++){
            array_[i] = 0.0;
        }
    }

    return;
}

/*====================================================================*/
RealArray::~RealArray() {
/*
    Destructor. Deletes the memory.
*/

    delete [] array_;

    return;
}

/*====================================================================*/
double& RealArray::operator [] (int index) {
/*
    Indexing is 0-based, like God intended it to be.
*/
    return(array_[index]);
}

/*====================================================================*/
int RealArray::size() const {
/*
    Query the size of this array.
*/
    return(size_);
}

/*====================================================================*/
void RealArray::size(int n) {
/*
    Set the size of this array to n. Re-allocates, and copies the
    contents of the existing array, if any, to the new array ONLY
    if the old array was smaller than the new one.
*/
    int i;

    if (size() == n) return;

    double *newArray = new double[n];
    if (!newArray){
        cout << "ERROR in RealArray::size(int): unable to allocate."
             << endl << flush;
        exit(0);
    }

    //copy the old data into the new array.
    if (size_ < n) {
        for(i=0; i<size_; i++){
            newArray[i] = array_[i];
        }
    }

    //initialize the rest of the array to zero.
    for(i=size_; i<n; i++){
        newArray[i] = 0.0;
    }

    //delete the old memory.
    delete [] array_;

    //point to the new memory.
    array_ = newArray;
    size_ = n;

    return;
}

/*====================================================================*/
void RealArray::resize(int n){
/*
    Does the same thing as ::size(int n).
*/
    size(n);

    return;
}

/*====================================================================*/
void RealArray::append(double item) {
/*
    Appends item to the end of array.
*/
    size(size()+1);

    array_[size()-1] = item;

    return;
}

/*====================================================================*/
void RealArray::insert(int index, double item) {
/*
    Inserts item in array at position index, moving data that was
    previously at position index, to position index+1.
*/

    if (index > size_) {
        cout << "ERROR in RealArray::insert: index > size of array"
             << endl << flush;
        exit(0);
    }

    int oldSize = size();

    //make the array bigger by 1
    size(oldSize+1);

    //now insert the new item after sliding everything else up.
    for(int i=oldSize; i>index; i--){
        array_[i] = array_[i-1];
    }
    array_[index] = item;
 
    return;
}

