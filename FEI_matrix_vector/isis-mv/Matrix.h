#ifndef __Matrix_H
#define __Matrix_H

//requires:
//#include "other/basicTypes.h" // needed for definition of bool
//#include "mv/Map.h"

class Vector;
class IntVector;

class Matrix  {
    
  public:
    
    // Constructor-type functions.
    Matrix(const Map& map);
    virtual ~Matrix() {};
    
    // Mathematical functions.
    virtual void vectorMultiply(Vector& y, const Vector& x) const = 0;
    virtual void transposeVectorMultiply(Vector& y, const Vector& x) const = 0;
    virtual void put(double s) = 0;
    
    // Testing functions.
    bool isFilled() const {return isFilled_;};
    bool isConfigured() const {return isConfigured_;};
    
    // Access functions...
    virtual void getDiagonal(Vector& diagVector) const = 0;
    const Map& getMap() const {return map_;};
    
    // Special functions.
    virtual void configure(const IntVector& rowCount) = 0;
    virtual void fillComplete() = 0;
    virtual bool readFromFile(char *fileName) = 0;
    virtual bool writeToFile(char *fileName) const = 0;

    // Min/Max functions
    virtual bool rowMax() const {return false;};
    virtual bool rowMin() const {return false;};
    virtual double rowMax(int rowNumber) const {return -rowNumber;};
    virtual double rowMin(int rowNumber) const {return -rowNumber;};
    
  protected:

    virtual void setFilled(bool flag)  {isFilled_ = flag;};
    virtual void setConfigured(bool flag)  {isConfigured_ = flag;};
    virtual bool rowScale(Vector& b) = 0;
    virtual bool colScale(Vector& colMax) = 0;
    friend class LinearEquations;

  protected:
    
    const Map& map_;    // map reference
    
  private:
    
    bool isFilled_;     // flag is true if matrix has been filled
    bool isConfigured_; // flag is true if matrix has been configured
    
};

#endif
