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
SIDLFortranSymbol(sidl_int__array_borrow_deref_f,
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
SIDLFortranSymbol(sidl_double__array_borrow_deref_f,
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


