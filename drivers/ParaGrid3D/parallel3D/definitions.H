#ifndef DEFINITIONS
#define DEFINITIONS


#include <vector.h>

/* using namespace std; */

//========================= D E F I N I T I O N S ============================

#define ON                1 
#define OFF               0

//============================================================================

#define TOLERANCE     10e-8

//============================================================================

#define Kmax             15  // used in gmres

//============================================================================

#define PRECONDITIONER    1  // 1 - no preconditioning
                             // 2 - HB preconditioning
                             // 3 - multigrid
#define MAINTAIN_MG     OFF
#define MAINTAIN_HB     OFF

//============================================================================

#define PORTNUM (14916+5000)     /* random port number, we need something */

#define OUTPUT_TYPE       1  // 0 - no output to file (or user defined)
                             // 1 - print the sol. on the bdr in MCGL format
                             // 2 - print the boundary mesh (maple format)
                             // 3 - print the mesh scaled (shrinked, maple)
                             // 4 - print a face with solution ( maple )
                             // 5 - print a cut face with the solution in mtv

#define OUTPUT_LM       OFF  // Print the local element matrices

//============================================================================

                             // Types of bdr (should be negative numbers).
#define INNER            -1  // Inner node (this applies only to nodes)
#define DIRICHLET        -2
#define NEUMANN          -3
#define ROBIN            -4

//============================================================================

#define TYPE_PU           1  // The following are types for the tetrahedrons
#define TYPE_A            2
#define TYPE_PF           3
#define TYPE_O            4
#define TYPE_M            5

//============================================================================

#define LEVEL            15  // Maximum number of Levels
 
//============================================================================

typedef int    three_int[3];
typedef int       int_16[16];
typedef double   double3[3];
typedef double   * p_double;
typedef double   *   p_real;
typedef double         real;
typedef double   *    reals;
typedef    int      idxtype;

typedef vector<int>   v_int;

//============================================================================

#define HYPRE_SOLVE      ON

//============================================================================

#define DOM_DECOMP       ON

#define PACKETS         336  // maximum number of packets
//#define MAX_PACKET     8192  // maximum number of nodes per packet
#define MAX_PACKET    20000
#define SHIFT            -5  // if face has tr<0 and < SHIFT then (SHIFT-tr)
                             // gives the number of the neighb. subdomain

#if     DOM_DECOMP  ==   ON
  #define MPI            ON  // ON if you want to use MPI

  #if     MPI       ==   ON   
     #include <mpi.h>
  #endif

#endif

//============================================================================

#define CONCENTRATION   OFF  // ON if we solve first pressure and then conc.

#if CONCENTRATION  ==    ON
  extern real convection[3]; // used for the velocity
  extern  int Pressure;      // 1 if pressure equations are being solved
#endif

//============================================================================

#define CONVECTION       ON
#define EXACT           OFF
#define PROBLEM           0  // 0 - for Reaction-Diffusion-Problem
                             // 1 - for Conv-Diff (discontinuous G approx.)
                             // 2 - for Conv-Diff (streamline-diffusion)
                             // 3 - Mixed Finite Element (RT0)

// Include the problem with the functions def. as given in "example.cpp"
#ifndef FUNCTIONS
#define FUNCTIONS
  #include "functions.h"
  //#include "bioscreen.cpp"          // bioscreen problem
  //#include "layers.cpp"
  //#include "convection_test.cpp" 
  //#include "pressure_bioscreen.cpp" // bioscreen problem without well-last
  //#include "test_mixed.cpp"         // test the mixed finite element code
  //#include "pressure_conc_well.cpp" // bioscreen problem with a well
  //#include "Lshape.cpp"
#endif  // FUNCTIONS

#define SILENT            ON

//============================================================================

#define WELL             OFF

//============================================================================

#endif


