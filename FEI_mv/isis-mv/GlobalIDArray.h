#ifndef __GlobalIDArray_h
#define __GlobalIDArray_h

/*
   This is a minimal int array class, which will provide the bare
   basics like self-description, allocation/re-sizing, inserting
   and appending. And maybe the [] operator.
*/

//requires:
//#include "other/basicTypes.h" //this include file contains the definition of
                        //'GlobalID'. Make sure it is correct at
                        //compile time.

class GlobalIDArray {
  public:
    //Constructors.
    GlobalIDArray(int n=0);

    //Destructor.
    virtual ~GlobalIDArray();

    //the functions...
    GlobalID& operator [] (int index);
    const GlobalID& operator [] (int index) const;
    int size() const;
    void size(int n);
    void resize(int n);
    void append(GlobalID item);
    void insert(int index, GlobalID item);

  private:
    GlobalID *array_; //the data.
    int size_;   //how much data there is.
    int internalSize_; //how big the actual memory is.
};
#endif

