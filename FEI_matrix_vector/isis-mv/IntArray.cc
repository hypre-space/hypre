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
#include "IntArray.h"

#ifndef MINGROW
#define MINGROW(b) (8 > (b) ? 8 : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

/*====================================================================*/
IntArray::IntArray(int n, int g, int init) {
/*
    Constructs array of size n.
*/

    if (n == 0) {
        array_ = NULL;
    }
    else {
        array_ = new int[n];
        if (array_ == NULL) {
            cout << "ERROR in IntArray(n,g): unable to allocate."
                << endl << flush;
            exit(0);
        }
    }
    internalSize_ = n;
    size_ = n;
    growSize_ = MINGROW(g);
    if (init) {
        fillfrom(0,0,internalSize_-1);
    }

    return;
}



/*====================================================================*/
IntArray::~IntArray() {
/*
    Destructor. Deletes the memory.
*/

    delete [] array_;

    return;
}

/*====================================================================*/
void IntArray::size(int n, int init) {
/*
    Set the size of this array to n. Re-allocates, and copies the
    contents of the existing array, if any, to the new array ONLY
    if the old array was smaller than the new one.
    inits new memory to 0. fixme.
*/
    // reduce size_ and return. internalSize_ unaffected
    if (size() >= n) {
        size_ = MAX(n,0); //ignore negative input
        return;
    }
    // expand size but only to  <=  internalSize_
    if (internalSize_ >= n) {
        size_ = n;
        return;
    }

    int *newArray;

    // make a bigger one and copy old data.
    if (n > internalSize_ + growSize_) {
        // grow to overgrown required
        newArray = new int[n];
        internalSize_ = n;
    }
    else {
        // grow by next growth increment.
        newArray = new int[internalSize_ + growSize_];
        internalSize_ += growSize_;
    }
    if (!newArray){
        cout << "ERROR in IntArray::size(int): unable to allocate."
             << endl << flush;
        exit(0);
    }

    // swap old,new memory.
    int* oldarray = array_;
    array_ = newArray;

    // preserve data from old memory
    copyfrom(0,size_,oldarray);

    //initialize the rest of the new array to zero.
    if (init) {
        fillfrom(0,size_,n-1);
    }

    //delete the old memory.
    delete [] oldarray;

    size_ = n;

    return;
}

/*====================================================================*/
void IntArray::resize(int n, int init){
/*
    Does the same thing as ::size(int n,int init).
*/
    size(n,init);

    return;
}

/*====================================================================*/
void IntArray::append(int item) {
/*
    Appends item to the end of array.
    Note that appending to 
    IntArray iaexample(10);
    appends elements starting at iaexample[10], not iaexample[0],
    because the constructor set the initial data.
*/
    size(size()+1,0); // with luck this usually amounts to incrementing size_

    array_[size()-1] = item;

    return;
}

/*====================================================================*/
void IntArray::insert(int index, int item) {
/*
    Inserts item in array at position index, moving data that was
    previously at position index, to position index+1.
*/

    if (size_==0){
       if (index==1) index = 0;
    }

    if (index > size_) {
        cout << "ERROR in IntArray::insert: index " << index 
             << " > size " << size_ << " of array" << endl << flush;
        abort();
    }

    int oldSize = size_;

    //make the array bigger by 1
    size(oldSize+1);

    //now insert the new item after sliding everything else up.
    int *ptr0 = &array_[oldSize], *ptr1 = &array_[oldSize-1];
    int *ptr2 = &array_[index];
//    for(int i=oldSize; i>index; i--){
//        array_[i] = array_[i-1];
//    }
//    array_[index] = item;

    while(ptr0>ptr2){
        *ptr0-- = *ptr1--;
    }
    *ptr0 = item;
 
    return;
}

/*====================================================================*/
void IntArray::fill(int item, int length) {
/*
    Fill first length values with item.
*/
    size(length,0);
    fillfrom(item,0,length-1);
    return;
}

/*====================================================================*/
// Private function -- no range checking.
void IntArray::fillfrom(int item, int low, int high) {
/* fill range [low..high] with item. */
    int i;
    for(i = low; i <= high; i++) {
        array_[i] = item;
    }
    return;
}

/*====================================================================*/
// Private function -- no range checking.
void IntArray::copyfrom(int start, int len, int *src) {
/* move data. src and this are assumed to be the correct sizes already. */
    int i;
    src += start;
    for (i = 0; i < len; i++) {
        array_[i] = src[i];
    }
    return;
}

/*====================================================================*/
void IntArray::copy(int start, int len, int *src) {
// public function

    // resize w/out initmem up to len
    size(len,0);

    // copy the data
    copyfrom(start,len,src);
}
