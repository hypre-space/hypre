#ifndef NUMBERING_DH_H
#define NUMBERING_DH_H

/* code and algorithms in this class adopted from Edmond Chow's
   ParaSails
*/

#include "euclid_common.h"

struct _numbering_dh {
  int   size;    /* max number of indices that can be stored */
  int   first;   /* global number of 1st index (row) */
  int   num_loc; /* number of local indices (number of rows in mat) */
  int   num_ext; /* number of external (non-local) indices; 
                  * total indices = num_loc+num_ext.
                  */
  int     *local_to_global;
  Hash_dh global_to_local;
};

extern void Numbering_dhCreate(Numbering_dh *numb);
extern void Numbering_dhDestroy(Numbering_dh numb);

  /* must be called before calling Numbering_dhGlobalToLocal() or
     Numbering_dhLocalToGlobal().
   */
extern void Numbering_dhSetup(Numbering_dh numb, Mat_dh mat);

  /* input: global_in[len], which contains global row numbers.
     output: local_out[len], containing corresponding local numbers.
     note: global_in[] and local_out[] may be identical.
   */
extern void Numbering_dhGlobalToLocal(Numbering_dh numb, int len, 
                                      int *global_in, int *local_out);


#endif
