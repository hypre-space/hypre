/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * LoadBal.h header file.
 *
 *****************************************************************************/

#ifndef _LOADBAL_H
#define _LOADBAL_H

#define LOADBAL_REQ_TAG  888
#define LOADBAL_REP_TAG  889

typedef struct
{
    int  pe;
    int  beg_row;
    int  end_row;
    int *buffer;
}
DonorData;

typedef struct
{
    int     pe;
    Matrix *mat;
    double *buffer;
}
RecipData;

void LoadBalDonate(MPI_Comm comm, ParaSails *ps, double local_cost, 
  double beta);
void LoadBalReturn(MPI_Comm comm, ParaSails *ps);

#endif /* _LOADBAL_H */
