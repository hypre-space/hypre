#ifndef VEC_DH_H
#define VEC_DH_H

#include "euclid_common.h"

struct _vec_dh {
  int n;
  double *vals;
};

extern void Vec_dhCreate(Vec_dh *v);
extern void Vec_dhDestroy(Vec_dh v);
extern void Vec_dhInit(Vec_dh v, int size);
        /* allocates storage, but does not initialize values */

extern void Vec_dhDuplicate(Vec_dh v, Vec_dh *out);
        /* creates vec and allocates storage, but neither
         * initializes nor copies values 
         */

extern void Vec_dhCopy(Vec_dh x, Vec_dh y);
        /* copies values from x to y;
         * y must have proper storage allocated,
         * e.g, through previous call to Vec_dhDuplicate,
         * or Vec_dhCreate and Vec_dhInit.
         */

extern void Vec_dhSet(Vec_dh v, double value);
extern void Vec_dhSetRand(Vec_dh v);

extern void Vec_dhPrint(Vec_dh v, FILE *fp);
extern void Vec_dhPrintToFile(Vec_dh v, char *filename);

#endif
