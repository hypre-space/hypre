/*--------------------------------------------------------------------------
 * Purpose:       Define data structure and access macros for incomplete
 *                factorization data, including the matrix, the factors of
 *                the matrix, and parameters used in the factorization and
 *                solve.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/

#ifndef _ILUDATA_HEADER
#define _ILUDATA_HEADER

#define ILUFact

#ifdef ICFact
  #define rscale scale
#endif

#include "matrix.h"
/*--------------------------------------------------------------------------
 * ilu_data
 *--------------------------------------------------------------------------*/

typedef struct
{

  int             mode;
  Matrix         *A;
  int             ipar[20];
  double          rpar[20];
  Matrix         *preconditioner;
  int             lenpmx;
  int            *perm;
  int            *inverse_perm;
  double         *rscale;
#ifdef ILUFact
  double         *cscale;
#endif
  int            *iwork;
  int             l_iwork;
  double         *rwork;
  int             l_rwork;
  
} ILUData;

/*--------------------------------------------------------------------------
 * Accessor functions for the ilu_data structure
 *--------------------------------------------------------------------------*/
#define ILUDataMode(ilu_data)               ((ilu_data) -> mode)
#define ILUDataA(ilu_data)                  ((ilu_data) -> A)
#define ILUDataIpar(ilu_data)               ((ilu_data) -> ipar)
#define ILUDataRpar(ilu_data)               ((ilu_data) -> rpar)
#define ILUDataPreconditioner(ilu_data)     ((ilu_data) -> preconditioner)
#define ILUDataLenpmx(ilu_data)             ((ilu_data) -> lenpmx)
#define ILUDataPerm(ilu_data)               ((ilu_data) -> perm)
#define ILUDataInversePerm(ilu_data)        ((ilu_data) -> inverse_perm)
#ifdef ILUFact
  #define ILUDataRscale(ilu_data)             ((ilu_data) -> rscale)
  #define ILUDataCscale(ilu_data)             ((ilu_data) -> cscale)
#else
  #define ILUDataScale(ilu_data)              ((ilu_data) -> scale)
  #define ILUDataRscale(ilu_data)             ILUDataScale(ilu_data)
  #define ILUDataCscale(ilu_data)             ILUDataScale(ilu_data)
#endif
#define ILUDataIwork(ilu_data)              ((ilu_data) -> iwork)
#define ILUDataLIwork(ilu_data)             ((ilu_data) -> l_iwork)
#define ILUDataRwork(ilu_data)              ((ilu_data) -> rwork)
#define ILUDataLRwork(ilu_data)             ((ilu_data) -> l_rwork)


#endif
