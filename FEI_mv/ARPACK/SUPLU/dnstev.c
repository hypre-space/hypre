#include <stdio.h>
#include <string.h>

void dnstev_(int *n,         int *nev,       char *which, 
             double *sigmar, double *sigmai, int *colptr,
             int *rowind,    double *nzvals, double *dr,
             double *di,     double *z,      int *ldz,
             int *info,      double *tol)
/*  Arguement list:

    n       (int*)    Dimension of the problem.  (INPUT)

    nev     (int*)    Number of eigenvalues requested.  (INPUT/OUTPUT)
                      This routine is used to compute NEV eigenvalues
                      nearest to a shift (sigmar, sigmai).
                      On return, it gives the number of converged
                      eigenvalues.

    which   (char*)   Specify which part of the spectrum is of interest.(INPUT)
                      which can be of the following type:
                      "LM"    --- eigenvalues with the largest magnitude.
                      "SM"    --- eigenvalues with the smallest magnitue.
                      "LR"    --- eigenvalues with the largest real part.
                      "SR"    --- eigenvalues with the smallest real part.
                      "LI"    --- eigenvalues with the largest imag part.
                      "SI"    --- eigenvalues with the largest imag part.
                      "shift" --- eigenvalues near a shift specified by
                                  (sigmar, sigmai).
                      Note:
                      Eigenvalues with the smallest magnitude can
                      be computed by setting which to 'shift' using
                      zero as the shift.

    sigmar  (double*) Real part of the shift. (INPUT)

    sigmai  (double*) Imaginar part of the shift. (INPUT)

    colptr  (int*)    dimension n+1. (INPUT)
                      Column pointers for the sparse matrix.

    rowind  (int*)    dimension colptr[*n]-1. (INPUT)
                      Row indices for the sparse matrix.

    nzvals  (double*) dimension colptr[*n]-1. (INPUT)
                      Nonzero values of the sparse matrix.
                      The sparse matrix is represented by
                      the above three arrays colptr, rowind, nzvals.

    dr      (double*) dimension nev+1.  (OUTPUT)
                      Real part of the eigenvalue.

    di      (double*) dimension nev+1.  (OUTPUT)
                      Imaginar part of the eigenvalue.

    z       (double*) dimension ldz by nev+1. (OUTPUT)
                      Eigenvector matrix.
                      If the j-th eigenvalue is real, the j-th column
                      of z contains the corresponding eigenvector.
                      If the j-th and j+1st eigenvalues form a complex
                      conjuagate pair, then the j-th column of z contains
                      the real part of the eigenvector, and the j+1st column
                      of z contains the imaginary part of the eigenvector.

    ldz      (int*)   The leading dimension of z. (INPUT)

    info     (int*)   Error flag to indicate whether the eigenvalues
                      calculation is successful. (OUTPUT)
                      *info = 0, successful exit
                      *info = 1, Maximum number of iteration is reached
                                 before all requested eigenvalues
                                 have converged.
*/
{
    *info = 0;

    if (strncmp(which,"shift",5) == 0 ||
        strncmp(which,"Shift",5) == 0 ||
        strncmp(which,"SHift",5) == 0 ||
        strncmp(which,"sHift",5) == 0 ) {
        
        dninst_(n,  nev, sigmar, sigmai, colptr, rowind, nzvals,
                dr, di, z, ldz, info, tol);
    }
    else if (strncmp(which,"LM",2) == 0 ||
             strncmp(which,"lm",2) == 0 || 
             strncmp(which,"Lm",2) == 0 ||
             strncmp(which,"lM",2) == 0) {
        which = "LM";
        dnexst_(n,  nev, which, colptr, rowind, nzvals,
                dr, di,  z,     ldz,    info);
    } 
    else if (strncmp(which,"SM",2) == 0 ||
             strncmp(which,"sm",2) == 0 || 
             strncmp(which,"Sm",2) == 0 ||
             strncmp(which,"sM",2) == 0) {
        which = "SM";
        dnexst_(n,  nev, which, colptr, rowind, nzvals,
                dr, di,  z,     ldz,    info);
    }
    else if (strncmp(which,"LR",2) == 0 ||
             strncmp(which,"lr",2) == 0 || 
             strncmp(which,"Lr",2) == 0 ||
             strncmp(which,"lR",2) == 0) {
        which = "LR";
        dnexst_(n,  nev, which, colptr, rowind, nzvals,
                dr, di,  z,     ldz,    info);
    }
    else if (strncmp(which,"SR",2) == 0 ||
             strncmp(which,"sr",2) == 0 || 
             strncmp(which,"Sr",2) == 0 ||
             strncmp(which,"sR",2) == 0 ) {
        which = "SR";
        dnexst_(n,  nev, which, colptr, rowind, nzvals,
                dr, di,  z,     ldz,    info);
    }
    else if (strncmp(which,"LI",2) == 0 ||
             strncmp(which,"li",2) == 0 || 
             strncmp(which,"Li",2) == 0 ||
             strncmp(which,"lI",2) == 0) {
        which = "LI";
        dnexst_(n,  nev, which, colptr, rowind, nzvals,
                dr, di,  z,     ldz,    info);
    }
    else if (strncmp(which,"SI",2) == 0 ||
             strncmp(which,"si",2) == 0 || 
             strncmp(which,"Si",2) == 0 ||
             strncmp(which,"sI",2) == 0 ) {
        which = "SI";
        dnexst_(n,  nev, which, colptr, rowind, nzvals,
                dr, di,  z,     ldz,    info);
    }
    else  {
        fprintf(stderr, "Invalid which, which must be one of the following:\n");
        fprintf(stderr, " Shift --- eigenvalues near a particular shift\n");
        fprintf(stderr, "           The shift must be provided through the\n");
        fprintf(stderr, "           arguements (sigmar, sigmai)\n"); 
        fprintf(stderr, " LM    --- eigenvalues with the largest magnitude\n");
        fprintf(stderr, " SM    --- eigenvalues with the smallest magnitude\n");
        fprintf(stderr, " LR    --- eigenvalues with the largest real part\n");
        fprintf(stderr, " SR    --- eigenvalues with the smallest real part\n");
        fprintf(stderr, " LI    --- eigenvalues with the largest imag part\n");
        fprintf(stderr, " SI    --- eigenvalues with the largest imag part\n");
        *info = -911;
    }
}
