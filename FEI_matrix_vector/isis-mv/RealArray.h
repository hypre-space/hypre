#ifndef __RealArray_h
#define __RealArray_h

/*
   This is a minimal double array class, which will provide the bare
   basics like self-description, allocation/re-sizing, inserting
   and appending. And maybe the [] operator.
*/

class RealArray {
  public:
    //Constructor.
    RealArray(int n=0);

    //Destructor.
    virtual ~RealArray();

    //the functions...
    double& operator [] (int index);
    int size() const;
    void size(int n);
    void resize(int n);
    void append(double item);
    void insert(int index, double item);

  private:
    double *array_; //the data.
    int size_;   //how much data there is.
};
#endif

