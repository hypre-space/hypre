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

#ifndef _AMG_FORTRAN_HEADER
#define _AMG_FORTRAN_HEADER


/* setup */
#define CALL_SETUP(Setup_err_flag, A, amg_data) \
setup_(&Setup_err_flag,\
       &AMGDataNumLevels(amg_data),\
       &AMGDataNSTR(amg_data),\
       &AMGDataECG(amg_data),\
       &AMGDataNCG(amg_data),\
       &AMGDataEWT(amg_data),\
       &AMGDataNWT(amg_data),\
       AMGDataICDep(amg_data),\
       &AMGDataIOutDat(amg_data),\
       &AMGDataNumUnknowns(amg_data),\
       AMGDataIMin(amg_data),\
       AMGDataIMax(amg_data),\
       MatrixData(A),\
       MatrixIA(A),\
       MatrixJA(A),\
       AMGDataIU(amg_data),\
       AMGDataIP(amg_data),\
       AMGDataICG(amg_data),\
       AMGDataIFG(amg_data),\
       MatrixData(AMGDataP(amg_data)),\
       MatrixIA(AMGDataP(amg_data)),\
       MatrixJA(AMGDataP(amg_data)),\
       AMGDataIPMN(amg_data),\
       AMGDataIPMX(amg_data),\
       AMGDataIV(amg_data),\
       AMGDataXP(amg_data),\
       AMGDataYP(amg_data),\
       &AMGDataNDIMU(amg_data),\
       &AMGDataNDIMP(amg_data),\
       &AMGDataNDIMA(amg_data),\
       &AMGDataNDIMB(amg_data),\
       AMGDataLogFileName(amg_data),\
       strlen(AMGDataLogFileName(amg_data)))

void setup_(int *, int *, int *, double *, int *, double *, int *,
	    int *, int *, int *, int *, int *,
	    double *, int *, int *,
	    int *, int *, int *, int *,
	    double *, int *, int *,
	    int *, int *, int *,
	    double *, double *,
            int *, int *, int *, int *,
	    char *, long);


/* idec */

void idec_(int *, int *, int *, int *);

/* rsdl */
#define CALL_RESIDUAL(energy,residual_nrm,resv,U_array,F_array,A_array,amg_data) \
rsdl_(&energy,\
      &residual_nrm,\
      &AMGDataNumUnknowns(amg_data),\
      resv,\
      AMGDataVecTemp(amg_data),\
      AMGDataIMin(amg_data),\
      AMGDataIMax(amg_data),\
      VectorData(U_array[0]),\
      VectorData(F_array[0]),\
      MatrixData(A_array[0]),\
      MatrixIA(A_array[0]),\
      MatrixJA(A_array[0]),\
      AMGDataIU(amg_data));

void rsdl_(double *, double *, int *, double *, double *, int *, int *,
           double *, double *, double *, int *, int *, int *);  

/* cycle */
#define CALL_CYCLE(Solve_err_flag, u, f, tol, amg_data) \
cycle_(&Solve_err_flag,\
       &AMGDataNumLevels(amg_data),\
       AMGDataMU(amg_data),\
       &Fcycle_flag,\
       &Vstar_flag,\
       AMGDataNTRLX(amg_data),\
       AMGDataIPRLX(amg_data),\
       AMGDataIERLX(amg_data),\
       AMGDataIURLX(amg_data),\
       &cycle_op_count,\
       &AMGDataNumUnknowns(amg_data),\
       AMGDataIMin(amg_data),\
       AMGDataIMax(amg_data),\
       VectorData(u),\
       VectorData(f),\
       AMGDataVecTemp(amg_data),\
       MatrixData(AMGDataA(amg_data)),\
       MatrixIA(AMGDataA(amg_data)),\
       MatrixJA(AMGDataA(amg_data)),\
       AMGDataIU(amg_data),\
       AMGDataICG(amg_data),\
       MatrixData(AMGDataP(amg_data)),\
       MatrixIA(AMGDataP(amg_data)),\
       MatrixJA(AMGDataP(amg_data)),\
       AMGDataIPMN(amg_data),\
       AMGDataIPMX(amg_data),\
       AMGDataIV(amg_data),\
       AMGDataIP(amg_data),\
       AMGDataXP(amg_data),\
       AMGDataYP(amg_data),\
       &AMGDataNDIMU(amg_data),\
       &AMGDataNDIMP(amg_data),\
       &AMGDataNDIMA(amg_data),\
       &AMGDataNDIMB(amg_data),\
       AMGDataLevA(amg_data),\
       AMGDataLevB(amg_data),\
       AMGDataLevV(amg_data),\
       AMGDataLevPI(amg_data),\
       AMGDataLevI(amg_data),\
       AMGDataNumA(amg_data),\
       AMGDataNumB(amg_data),\
       AMGDataNumV(amg_data),\
       AMGDataNumP(amg_data));

void cycle_(int *, int *, int *, int *, int *, int *, int *, 
            int *, int *,
	    int *, int *, int *, int *,
	    double *, double *, double *,
	    double *, int *, int *,
	    int *, int *,
	    double *, int *, int *,
	    int *, int *, int *, int *,
	    double *, double *,
            int *, int *, int *, int *,
            int *, int *, int *, int *, int *,
            int *, int *, int *, int *);



/* relax  */
#define CALL_RELAX(Solve_err_flag, u, f, tol, amg_data) \
relax_(&Solve_err_flag,\
       &ity[k], &ipt[k], &ieq[k], &iun[k],\
       &imin[level],&imax[level],\
       VectorData(U_array[level]),\
       VectorData(F_array[level]),\
       MatrixData(A_array[level]),\
       MatrixIA(A_array[level]),\
       MatrixJA(A_array[level]),\
       VectorIntData(IU_array[level]),\
       VectorIntData(ICG_array[level]),\
       &ipmn[level],\
       &ipmx[level],\
       VectorIntData(IV_array[level]));

 
void relax_(int *, int *, int *, int *, int *, int *, int *, 
	    double *, double *, double *,
	    int *, int *, int *, int *, int *, int *, int *);



/* intad */
#define CALL_INTAD(coarse_grid,fine_grid,numv,Vstar_flag,F_array,U_array,amg_data) \
intad_(VectorData(U_array[fine_grid]),\
       MatrixData(P_array[fine_grid]),\
       MatrixIA(P_array[fine_grid]),\
       MatrixJA(P_array[fine_grid]),\
       &numv[fine_grid],\
       AMGDataVecTemp(amg_data),\
       VectorData(U_array[coarse_grid]),\
       VectorData(F_array[coarse_grid]),\
       MatrixData(A_array[coarse_grid]),\
       MatrixIA(A_array[coarse_grid]),\
       MatrixJA(A_array[coarse_grid]),\
       VectorIntData(IU_array[coarse_grid]),\
       &numv[coarse_grid],\
       &Vstar_flag,\
       &AMGDataNumUnknowns(amg_data));


void intad_(double *, double *, int *, int *, int *,
            double *, double *, double *, double *, 
            int *, int *, int *, int *, int *, int *);	



/* rscali */ 
#define CALL_RSCALI(coarse_grid,fine_grid,numv,F_array,U_array,amg_data) \
rscali_(VectorData(F_array[coarse_grid]),\
        &numv[coarse_grid],\
        VectorData(U_array[fine_grid]),\
        VectorData(F_array[fine_grid]),\
        AMGDataVecTemp(amg_data),\
        MatrixData(A_array[fine_grid]),\
        MatrixIA(A_array[fine_grid]),\
        MatrixJA(A_array[fine_grid]),\
        MatrixData(P_array[fine_grid]),\
        MatrixIA(P_array[fine_grid]),\
        MatrixJA(P_array[fine_grid]),\
        &numv[fine_grid]);                   

void rscali_(double *, int *, double *, double *, double *,
             double *, int *, int *, 
             double *, int *, int *, int *); 
   




#endif
