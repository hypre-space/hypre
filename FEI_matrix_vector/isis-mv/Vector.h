#ifndef __Vector_H
#define __Vector_H

//requires:
//#include <assert.h>
//#include <math.h>
//#include <unistd.h>
//#include <stdlib.h>
//#include <iostream.h>
//#include <string.h>
//#include <mpi.h>
//#include "other/basicTypes.h"
//#include "mv/CommInfo.h"
//#include "mv/Map.h"

/**=========================================================================**/
class Vector  
{
  public:
    // Constructor-type functions.
    Vector(const Map& map) : map_(map) {};
    virtual ~Vector() {};
    virtual Vector* newVector() const = 0;

    // Mathematical functions.
    virtual double dotProd(const Vector& y) const = 0;
    virtual void scale(double s) = 0;
    virtual void linComb(const Vector& b, double s, const Vector& c) = 0;
    virtual void addVec(double s, const Vector& c) = 0;
    virtual double norm() const = 0; // sqrt(sum(xi*xi))
    virtual double norm1() const = 0; // max abs(xi)
    virtual void random(int seed=1) = 0;
    
    // operator=
    Vector& operator = (const Vector& rhs);
    
    // Access functions.
    virtual double& operator [] (int index) = 0;
    virtual const double& operator [] (int index) const = 0;

    const Map& getMap() const {return map_;};

    virtual void put(double scalar) = 0;
    
    //Special functions
    virtual bool readFromFile(char *fileName) = 0;
    virtual bool writeToFile(char *fileName) const = 0;
    
    // emulated RTTI
    virtual char* myType() const {return "Vector";};
    virtual bool isDynCastToOK(const Vector& trial) const {
        if (!strcmp(myType(),trial.myType()))
            return(true);
        else return(false);
    };
    
  protected:
    virtual void assign(const Vector& rhs);
    virtual double QMR3norm2dot(const Vector& wtilde,
        const Vector& vtilde, const Vector& tmp,
        double& gamma, double& xi, double& rho, double& epsilon) 
        const = 0; 
    friend class QMR2_Solver;
    
  protected:
    virtual const double* startPointer() const = 0;
    friend class Aztec_Vector;
    friend class AztecDMSR_Matrix;
    friend class AztecDVBR_Matrix;
    friend class BLU_PC;
 
  protected:
    const Map& map_;            // local copy of map
    
};

#endif

