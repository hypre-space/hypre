/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * C/Fortran interface macros
 *
 *****************************************************************************/

#include "amgs01.h"

/* setup */
#define CALL_SETUP(problem, amgs01_data) \
setup_(&AMGS01DataNumLevels(amgs01_data),\
       &AMGS01DataNSTR(amgs01_data),\
       &AMGS01DataECG(amgs01_data),\
       &AMGS01DataNCG(amgs01_data),\
       &AMGS01DataEWT(amgs01_data),\
       &AMGS01DataNWT(amgs01_data),\
       AMGS01DataICDep(amgs01_data),\
       &AMGS01DataIOutMat(amgs01_data),\
       &ProblemNumUnknowns(problem),\
       AMGS01DataIMin(amgs01_data),\
       AMGS01DataIMax(amgs01_data),\
       VectorData(ProblemU(problem)),\
       VectorData(ProblemF(problem)),\
       MatrixData(AMGS01DataA(amgs01_data)),\
       MatrixIA(AMGS01DataA(amgs01_data)),\
       MatrixJA(AMGS01DataA(amgs01_data)),\
       ProblemIU(problem),\
       ProblemIP(problem),\
       AMGS01DataICG(amgs01_data),\
       AMGS01DataIFG(amgs01_data),\
       MatrixData(AMGS01DataP(amgs01_data)),\
       MatrixIA(AMGS01DataP(amgs01_data)),\
       MatrixJA(AMGS01DataP(amgs01_data)),\
       AMGS01DataIPMN(amgs01_data),\
       AMGS01DataIPMX(amgs01_data),\
       ProblemIV(problem),\
       ProblemXP(problem),\
       ProblemYP(problem),\
       &AMGS01DataNDIMU(amgs01_data),\
       &AMGS01DataNDIMP(amgs01_data),\
       &AMGS01DataNDIMA(amgs01_data),\
       &AMGS01DataNDIMB(amgs01_data),\
       AMGS01DataLogFileName(amgs01_data),\
       strlen(AMGS01DataLogFileName(amgs01_data)))

void setup_(int *, int *, double *, int *, double *, int *,
	    int *, int *, int *, int *, int *,
	    double *, double *,
	    double *, int *, int *,
	    int *, int *, int *, int *,
	    double *, int *, int *,
	    int *, int *, int *,
	    double *, double *,
            int *, int *, int *, int *,
	    char *, long);

/* solve */
#define CALL_SOLVE(u, f, tol, amgs01_data) \
{\
   Problem  *problem = AMGS01DataProblem(amgs01_data);\
   solve_(&AMGS01DataNumLevels(amgs01_data),\
	  &AMGS01DataNCyc(amgs01_data),\
	  AMGS01DataMU(amgs01_data),\
	  AMGS01DataNTRLX(amgs01_data),\
	  AMGS01DataIPRLX(amgs01_data),\
	  AMGS01DataIERLX(amgs01_data),\
	  AMGS01DataIURLX(amgs01_data),\
	  &AMGS01DataIOutRes(amgs01_data),\
	  &ProblemNumUnknowns(problem),\
	  AMGS01DataIMin(amgs01_data),\
	  AMGS01DataIMax(amgs01_data),\
	  VectorData(u),\
	  VectorData(f),\
	  MatrixData(AMGS01DataA(amgs01_data)),\
	  MatrixIA(AMGS01DataA(amgs01_data)),\
	  MatrixJA(AMGS01DataA(amgs01_data)),\
	  ProblemIU(problem),\
	  AMGS01DataICG(amgs01_data),\
	  MatrixData(AMGS01DataP(amgs01_data)),\
	  MatrixIA(AMGS01DataP(amgs01_data)),\
	  MatrixJA(AMGS01DataP(amgs01_data)),\
	  AMGS01DataIPMN(amgs01_data),\
	  AMGS01DataIPMX(amgs01_data),\
	  ProblemIV(problem),\
	  ProblemIP(problem),\
	  ProblemXP(problem),\
	  ProblemYP(problem),\
          &AMGS01DataNDIMU(amgs01_data),\
          &AMGS01DataNDIMP(amgs01_data),\
          &AMGS01DataNDIMA(amgs01_data),\
          &AMGS01DataNDIMB(amgs01_data),\
	  AMGS01DataLevA(amgs01_data),\
	  AMGS01DataLevB(amgs01_data),\
	  AMGS01DataLevV(amgs01_data),\
	  AMGS01DataLevPI(amgs01_data),\
	  AMGS01DataLevI(amgs01_data),\
	  AMGS01DataNumA(amgs01_data),\
	  AMGS01DataNumB(amgs01_data),\
	  AMGS01DataNumV(amgs01_data),\
	  AMGS01DataNumP(amgs01_data),\
	  AMGS01DataLogFileName(amgs01_data),\
	  strlen(AMGS01DataLogFileName(amgs01_data)));\
}

void solve_(int *, int *, int *, int *, int *, int *, int *,
	    int *, int *, int *, int *,
	    double *, double *,
	    double *, int *, int *,
	    int *, int *,
	    double *, int *, int *,
	    int *, int *, int *, int *,
	    double *, double *,
            int *, int *, int *, int *,
            int *, int *, int *, int *, int *,
            int *, int *, int *, int *,
	    char *, long);

