#ifndef _WELL_
#define _WELL_

#include "definitions.h" 
#include "Method.h" 

//============================================================================
// The well charachteristics are given by alpha and Q. When we solve for the
// pressure we have : -K dp/dn = alpha(p - p_0) = alpha p - Q .
// The well is modeled as line starting with z = z_start and ending at 
// z = z_end. Function is_well (pointer to function which is passed) determi-
// nes whether or not the point belongs to the well or not - the decision is
// besed on node's atribut and its coordinates. The function is passed as
// one of the arguments of the well constructor.
// We add the contributions to the stiffness matrix and the RHS b (function
// add_contribution) by going through the well's nodes from z_start to z_end
// (using function get_next) and adding the corresponding contributions.  
//============================================================================

class Well{

  protected :
    Method *m;           // Pointer to the mesh

    double z_start, z_end;
    double alpha;
    double Q;

    double *pressure;    // If pressure != NULL this is well for the concen-
                         // tration.

    int (*is_well)(int atr, real *coord); 
    int get_next(int level, int ind1, int &ind2, double z1, double &z2);

  public :
    Well(Method *mpointer, double zs, double ze, double alpha, double Q,
	 int (*)(int, real *));
    Well(Method *mpointer, double zs, double ze, double alpha, double Q,
	 int (*)(int, real *), double *pr);

    void add_contribution(int level, real *A, double *b);
};

#endif
