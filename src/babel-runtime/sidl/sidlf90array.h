/*
 * File:        sidlf90array.h
 * Copyright:   (c) 2003 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.8 $
 * Date:        $Date: 2007/09/27 19:35:48 $
 * Description: Functions to convert sidl arrays into F90 derived type
 *
 */

#ifndef included_sidlf90array_h
#define included_sidlf90array_h

#ifndef included_sidlType_h
#include "sidlType.h"
#endif

#if defined(SIDL_MAX_F90_DESCRIPTOR) && !defined(FORTRAN90_DISABLED)

/* forward declaration of sidl array struct's */
struct sidl_dcomplex__array;
struct sidl_double__array;
struct sidl_fcomplex__array;
struct sidl_float__array;
struct sidl_int__array;
struct sidl_long__array;

struct sidl_fortran_array {
  int64_t d_ior;
  char    d_descriptor[SIDL_MAX_F90_DESCRIPTOR];
};

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Convert a sidl IOR into a F90 derived type containing the IOR pointer
 * as a 64 bit integer and a F90 pointer to an array.
 * src        NULL or a valid sidl array
 * src_dimen  the dimension of src (*only* used when src is NULL)
 * desc       must be non-NULL pointer to single struct. This incoming
 *            contents of this struct are ignored. The incoming contents
 *            are overwritten.
 *
 * return value 0 means everything worked. Non-zero means it failed
 */
int
sidl_dcomplex__array_convert2f90(const struct sidl_dcomplex__array *src,
                                 const int src_dimen,
                                 struct sidl_fortran_array *dest);


/**
 * Convert a sidl IOR into a F90 derived type containing the IOR pointer
 * as a 64 bit integer and a F90 pointer to an array.
 * src        NULL or a valid sidl array
 * src_dimen  the dimension of src (*only* used when src is NULL)
 * desc       must be non-NULL pointer to single struct. This incoming
 *            contents of this struct are ignored. The incoming contents
 *            are overwritten.
 *
 * return value 0 means everything worked. Non-zero means it failed
 */
int
sidl_double__array_convert2f90(const struct sidl_double__array *src,
                               const int src_dimen,
                               struct sidl_fortran_array *dest);

/**
 * Convert a sidl IOR into a F90 derived type containing the IOR pointer
 * as a 64 bit integer and a F90 pointer to an array.
 * src        NULL or a valid sidl array
 * src_dimen  the dimension of src (*only* used when src is NULL)
 * desc       must be non-NULL pointer to single struct. This incoming
 *            contents of this struct are ignored. The incoming contents
 *            are overwritten.
 *
 * return value 0 means everything worked. Non-zero means it failed
 */
int
sidl_fcomplex__array_convert2f90(const struct sidl_fcomplex__array *src,
                                 const int src_dimen,
                                 struct sidl_fortran_array *dest);

/**
 * Convert a sidl IOR into a F90 derived type containing the IOR pointer
 * as a 64 bit integer and a F90 pointer to an array.
 * src        NULL or a valid sidl array
 * src_dimen  the dimension of src (*only* used when src is NULL)
 * desc       must be non-NULL pointer to single struct. This incoming
 *            contents of this struct are ignored. The incoming contents
 *            are overwritten.
 *
 * return value 0 means everything worked. Non-zero means it failed
 */
int
sidl_float__array_convert2f90(const struct sidl_float__array *src,
                              const int src_dimen,
                              struct sidl_fortran_array *dest);

/**
 * Convert a sidl IOR into a F90 derived type containing the IOR pointer
 * as a 64 bit integer and a F90 pointer to an array.
 * src        NULL or a valid sidl array
 * src_dimen  the dimension of src (*only* used when src is NULL)
 * desc       must be non-NULL pointer to single struct. This incoming
 *            contents of this struct are ignored. The incoming contents
 *            are overwritten.
 *
 * return value 0 means everything worked. Non-zero means it failed
 */
int
sidl_int__array_convert2f90(const struct sidl_int__array *src,
                            const int src_dimen,
                            struct sidl_fortran_array *dest);

/**
 * Convert a sidl IOR into a F90 derived type containing the IOR pointer
 * as a 64 bit integer and a F90 pointer to an array.
 * src        NULL or a valid sidl array
 * src_dimen  the dimension of src (*only* used when src is NULL)
 * desc       must be non-NULL pointer to single struct. This incoming
 *            contents of this struct are ignored. The incoming contents
 *            are overwritten.
 *
 * return value 0 means everything worked. Non-zero means it failed
 */
int
sidl_long__array_convert2f90(const struct sidl_long__array *src,
                             const int src_dimen,
                             struct sidl_fortran_array *dest);
#ifdef __cplusplus
}
#endif


#endif /* defined(SIDL_MAX_F90_DESCRIPTOR) && !defined(FORTRAN90_DISABLED) */
#endif /*  included_sidlf90array_h */
