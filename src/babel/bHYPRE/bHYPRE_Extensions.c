#include "SIDL_header.h"
#include "babel_config.h"
#include "SIDLfortran.h"
#include "SIDLType.h"
#include "SIDLArray.h"
#include <stdlib.h>
#include <stddef.h>

/* same as SIDL_int__array_borrow_f, but the first argument is a pointer
   that gets dereferenced */
void
SIDLFortran77Symbol(sidl_int__array_borrow_deref_f,
                  SIDL_INT__ARRAY_BORROW_DEREF_F,
                  SIDL_int__array_borrow_deref_f)
  (int32_t **firstElement, int32_t *dimen, int32_t lower[], int32_t upper[], int32_t stride[], int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_int__array_borrow(*firstElement, 
                               *dimen,
                               lower,
                               upper,
                               stride);
}


/* same as SIDL_double__array_borrow_f, but the first argument is a pointer
   that gets dereferenced */
void
SIDLFortran77Symbol(sidl_double__array_borrow_deref_f,
                  SIDL_DOUBLE__ARRAY_BORROW_DEREF_F,
                  SIDL_double__array_borrow_deref_f)
  (double **firstElement, int32_t *dimen, int32_t lower[], int32_t upper[], int32_t stride[], int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_double__array_borrow(*firstElement, 
                               *dimen,
                               lower,
                               upper,
                               stride);
}


/* same as SIDL_int__array_set1_f, but the third argument is, not the value,
   but a pointer to an array; and there is a fourth argument, the index to the array.
    */
void
SIDLFortran77Symbol(sidl_int__array_set1_deref_f,
                  SIDL_INT__ARRAY_SET1_DEREF_F,
                  SIDL_int__array_set1_deref_f)
  (int64_t *array,
   int32_t *i1,
   int32_t ***value,
   int32_t *i2)
{
  SIDL_int__array_set1((struct SIDL_int__array *)(ptrdiff_t)*array,
   *i1,
   (**value)[*i2]);
}



