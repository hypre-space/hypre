
#ifndef _BJ_DATA_HEADER
#define _BJ_DATA_HEADER


/* Include Petsc linear solver headers */
#include "sles.h"

#include <stdio.h>

#include "matrix.h"

#define  NDIMU(nv)  (nv)
#define  NDIMP(np)  (np)
#define  NDIMA(na)  (na)
#define  NDIMB(na)  (na)

#include "general.h"

/* Include prototypes */
#include "PETSc_BP.h"


/*--------------------------------------------------------------------------
 * BJData
 *--------------------------------------------------------------------------*/


typedef struct
{

  Matrix         *A;
  int             A_is_true; /* 1 if A is not modified, or is replaced,
                                between setup and solve */

  /* Linear solver structure from Petsc */
  SLES           *sles_ptr;

  /* Information to feed into the local solver routine */
  void           *ls_data;

} BJData;

/*--------------------------------------------------------------------------
 * Accessor functions for the BJData structure
 *--------------------------------------------------------------------------*/

#define BJDataA(bj_data)               ((bj_data) -> A)
#define BJDataA_is_true(bj_data)       ((bj_data) -> A_is_true)
#define BJDataSles_ptr(bj_data)        ((bj_data) -> sles_ptr)
#define BJDataLsData(bj_data)          ((bj_data) -> ls_data)

#endif
