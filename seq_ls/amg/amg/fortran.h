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
 * C to Fortran interfacing macros
 *
 *****************************************************************************/

/* amgs01.f */
#define amgs01(u, f, A, problem, ifc, isw, log_file_name) \
amgs01_(VectorData(u), VectorData(f),\
	MatrixData(A), MatrixIA(A), MatrixJA(A),\
	&ProblemNumVariables(problem),\
	&ProblemNumUnknowns(problem),\
	&ProblemNumPoints(problem),\
	ProblemIU(problem), ProblemIP(problem), ProblemIV(problem),\
	ProblemX(problem), ProblemY(problem), ProblemZ(problem),\
	ifc, &isw, log_file_name,\
	strlen(log_file_name))

void amgs01_(double *u, double *f,
	     double *a, int *ia, int *ja,
	     int *nv, int *nu, int *np,
	     int *iu, int *ip, int *iv,
	     double *x, double *y, double *z,
	     int *ifc, int *isw, char *log_file_name,
	     long log_file_name_len);

