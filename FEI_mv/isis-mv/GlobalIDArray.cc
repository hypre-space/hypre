#include <stdio.h>
#include <stdlib.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "GlobalIDArray.h"

/*====================================================================*/
GlobalIDArray::GlobalIDArray(int n) {
/*
    Constructs array of size n, initializes its contents to 0.
*/

    if (n == 0) {
        array_ = NULL;
        size_ = 0;
        internalSize_ = 0;
    }
    else {
        array_ = new GlobalID[n];
        size_ = n;
        internalSize_ = 0;

        for(int i=0; i<size_; i++){
            array_[i] = (GlobalID)0;
        }
    }

    return;
}

/*====================================================================*/
GlobalIDArray::~GlobalIDArray() {
/*
    Destructor. Deletes the memory.
*/

    delete [] array_;

    return;
}

/*====================================================================*/
GlobalID& GlobalIDArray::operator [] (int index) {
/*
    Indexing is 0-based, like God intended it to be.
*/
    return(array_[index]);
}
 
/*====================================================================*/
const GlobalID& GlobalIDArray::operator [] (int index) const {
/*
    Indexing is 0-based, like God intended it to be.
*/
    return(array_[index]);
}
 
/*====================================================================*/
int GlobalIDArray::size() const {
/*
    Query the size of this array.
*/
    return(size_);
}

/*====================================================================*/
void GlobalIDArray::size(int n) {
/*
    Set the size of this array to n. Re-allocates, and copies the
    contents of the existing array, if any, to the new array ONLY
    if the old array was smaller than the new one.
*/
    int i;

    if (size() == n) return;

    int *newArray;

    if (n >= internalSize_){
        newArray = new int[n+100];
        internalSize_ = n+100;
    }
    else {
        size_ = n;
        return;
    }

    if (!newArray){
        cout << "ERROR in GlobalIDArray::size(int): unable to allocate."
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
        newArray[i] = 0;
    }

    //delete the old memory.
    delete [] array_;

    //point to the new memory.
    array_ = newArray;
    size_ = n;

    return;
}

/*====================================================================*/
void GlobalIDArray::resize(int n){
/*
    Does the same thing as ::size(int n).
*/
    size(n);

    return;
}

/*====================================================================*/
void GlobalIDArray::append(GlobalID item) {
/*
    Appends item to the end of array.
*/
    size(size()+1);

    array_[size()-1] = item;

    return;
}

/*====================================================================*/
void GlobalIDArray::insert(int index, GlobalID item) {
/*
    Inserts item in array at position index, moving data that was
    previously at position index, to position index+1.
*/

    if (index > size_) {
        cout << "ERROR in GlobalIDArray::insert: index > size of array"
             << endl << flush;
        abort();
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

