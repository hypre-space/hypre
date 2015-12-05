/*
 * File:        sidlfortran.h
 * Revision:    @(#) $Revision: 1.7 $
 * Date:        $Date: 2006/08/29 22:29:52 $
 * Description: Macros for FORTRAN interoperability
 *
 * Copyright (c) 2000-2002, The Regents of the University of Calfornia.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * UCRL-CODE-2002-054
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#ifndef included_sidlfortran_h
#define included_sidlfortran_h

#ifndef included_babel_config_h
#include "babel_config.h"
#endif
#ifdef FORTRAN77_DISABLED
#ifdef FORTRAN90_DISABLED
#error This installation of Babel Runtime was configured without any Fortran support.
#endif
#endif


#ifndef FORTRAN77_DISABLED
/*
 * The SIDLFortran77Ending macro should be defined to add the appropriate
 * characters to the end of a symbol.  Normally, this means adding zero, 
 * one or two underscores.
 */
#ifdef SIDL_F77_ZERO_UNDERSCORE
#define SIDLFortran77Ending(sym) sym
#else
#ifdef SIDL_F77_ONE_UNDERSCORE
#define SIDLFortran77Ending(sym) sym ## _
#else
#ifdef SIDL_F77_TWO_UNDERSCORE
#define SIDLFortran77Ending(sym) sym ## __
#else
#error one of SIDL_F77_(ZERO,ONE,TWO)_UNDERSCORE must be defined
#endif
#endif
#endif

/*
 * The SIDLFortran77Symbol macro should be defined to choose between
 * a lower case form of the symbol, an upper case form of the symbol and
 * a mixed case form of the symbol.  It should apply SIDLFortran77Ending
 * to its choice.
 */
#ifdef SIDL_F77_LOWER_CASE
#define SIDLFortran77Symbol(lcase,ucase,mcase) SIDLFortran77Ending(lcase)
#else
#ifdef SIDL_F77_UPPER_CASE
#define SIDLFortran77Symbol(lcase,ucase,mcase) SIDLFortran77Ending(ucase)
#else
#ifdef SIDL_F77_MIXED_CASE
#define SIDLFortran77Symbol(lcase,ucase,mcase) SIDLFortran77Ending(mcase)
#else
#error one of SIDL_F77_(LOWER,UPPER,MIXED)_CASE must be defined
#endif
#endif
#endif

/*
 * String handling
 */
#if defined(SIDL_F77_STR_LEN_NEAR) || defined(SIDL_F77_STR_LEN_FAR)
typedef char *SIDL_F77_String;
#define SIDL_F77_STR(strvar) (strvar)
#define SIDL_F77_STR_LOCAL_ARG(strvar) strvar
#define SIDL_F77_STR_LOCAL_STR(strvar) strvar
#define SIDL_F77_STR_LOCAL_LEN(strvar) strvar ## _len
#define SIDL_F77_STR_LEN(strvar) strvar ## _len
#define SIDL_F77_STR_LOCAL(strvar) \
SIDL_F77_String strvar = NULL;\
int SIDL_F77_STR_LOCAL_LEN(strvar) = 0
#ifdef SIDL_F77_STR_LEN_NEAR
#define SIDL_F77_STR_NEAR_LEN_DECL(strvar) , int SIDL_F77_STR_LEN(strvar)
#define SIDL_F77_STR_NEAR_LEN(strvar) , SIDL_F77_STR_LOCAL_LEN(strvar)
#define SIDL_F77_STR_FAR_LEN_DECL(strvar) 
#define SIDL_F77_STR_FAR_LEN(strvar)
#else
#define SIDL_F77_STR_NEAR_LEN_DECL(strvar)
#define SIDL_F77_STR_NEAR_LEN(strvar)
#define SIDL_F77_STR_FAR_LEN_DECL(strvar) , int SIDL_F77_STR_LEN(strvar)
#define SIDL_F77_STR_FAR_LEN(strvar) , SIDL_F77_STR_LOCAL_LEN(strvar)
#endif
#else
#if defined(SIDL_F77_STR_STRUCT_STR_LEN) || defined(SIDL_F77_STR_STRUCT_LEN_STR)
struct SIDL_F77_String_t;
typdef struct SIDL_F77_String_t *SIDL_F77_String;
#define SIDL_F77_STR(strvar) ((*strvar).str)
#define SIDL_F77_STR_LOCAL(strvar) struct SIDL_F77_String_t strvar = \
  { NULL, 0 }
#define SIDL_F77_STR_LOCAL_ARG(strvar) &strvar
#define SIDL_F77_STR_LOCAL_STR(strvar) ((strvar).str)
#define SIDL_F77_STR_LOCAL_LEN(strvar) ((strvar).len)
#define SIDL_F77_STR_LEN(strvar) ((*strvar).len)
#define SIDL_F77_STR_NEAR_LEN_DECL(strvar)
#define SIDL_F77_STR_FAR_LEN_DECL(strvar)
#else
#ifdef SIDL_F77_STR_STRUCT_STR_LEN
struct SIDL_F77_String_t {
  char *str;
  int  len;
};
#else
struct SIDL_F77_String_t {
  int  len;
  char *str;
};
#endif
#endif
#endif
#define SIDL_F77_STR_COPY(localvar,argvar,minsize) \
  sidl_copy_ior_str(&(SIDL_F77_STR_LOCAL_STR(localvar)), \
                    &(SIDL_F77_STR_LOCAL_LEN(localvar)), \
                    (argvar), (minsize))

#ifndef SIDL_F77_STR_MINSIZE
#define SIDL_F77_STR_MINSIZE 512
#endif

typedef int SIDL_F77_Bool;
#ifndef SIDL_F77_TRUE
#define SIDL_F77_TRUE 1
#endif
#ifndef SIDL_F77_FALSE 
#define SIDL_F77_FALSE 0
#endif

#endif /* not FORTRAN77_DISABLED */


#ifndef FORTRAN90_DISABLED
/*
 * The SIDLFortran90Ending macro should be defined to add the appropriate
 * characters to the end of a symbol.  Normally, this means adding zero, 
 * one or two underscores.
 */
#ifdef SIDL_F90_ZERO_UNDERSCORE
#define SIDLFortran90Ending(sym) sym
#else
#ifdef SIDL_F90_ONE_UNDERSCORE
#define SIDLFortran90Ending(sym) sym ## _
#else
#ifdef SIDL_F90_TWO_UNDERSCORE
#define SIDLFortran90Ending(sym) sym ## __
#else
#error one of SIDL_F90_(ZERO,ONE,TWO)_UNDERSCORE must be defined
#endif
#endif
#endif

/*
 * The SIDLFortran90Symbol macro should be defined to choose between
 * a lower case form of the symbol, an upper case form of the symbol and
 * a mixed case form of the symbol.  It should apply SIDLFortran90Ending
 * to its choice.
 */
#ifdef SIDL_F90_LOWER_CASE
#define SIDLFortran90Symbol(lcase,ucase,mcase) SIDLFortran90Ending(lcase)
#else
#ifdef SIDL_F90_UPPER_CASE
#define SIDLFortran90Symbol(lcase,ucase,mcase) SIDLFortran90Ending(ucase)
#else
#ifdef SIDL_F90_MIXED_CASE
#define SIDLFortran90Symbol(lcase,ucase,mcase) SIDLFortran90Ending(mcase)
#else
#error one of SIDL_F90_(LOWER,UPPER,MIXED)_CASE must be defined
#endif
#endif
#endif

/*
 * String handling
 */
#if defined(SIDL_F90_STR_LEN_NEAR) || defined(SIDL_F90_STR_LEN_FAR)
typedef char *SIDL_F90_String;
#define SIDL_F90_STR(strvar) (strvar)
#define SIDL_F90_STR_LOCAL_ARG(strvar) strvar
#define SIDL_F90_STR_LOCAL_STR(strvar) strvar
#define SIDL_F90_STR_LOCAL_LEN(strvar) strvar ## _len
#define SIDL_F90_STR_LEN(strvar) strvar ## _len
#define SIDL_F90_STR_LOCAL(strvar) \
SIDL_F90_String strvar = NULL;\
int SIDL_F90_STR_LOCAL_LEN(strvar) = 0
#ifdef SIDL_F90_STR_LEN_NEAR
#define SIDL_F90_STR_NEAR_LEN_DECL(strvar) , int SIDL_F90_STR_LEN(strvar)
#define SIDL_F90_STR_NEAR_LEN(strvar) , SIDL_F90_STR_LOCAL_LEN(strvar)
#define SIDL_F90_STR_FAR_LEN_DECL(strvar) 
#define SIDL_F90_STR_FAR_LEN(strvar)
#else
#define SIDL_F90_STR_NEAR_LEN_DECL(strvar)
#define SIDL_F90_STR_NEAR_LEN(strvar)
#define SIDL_F90_STR_FAR_LEN_DECL(strvar) , int SIDL_F90_STR_LEN(strvar)
#define SIDL_F90_STR_FAR_LEN(strvar) , SIDL_F90_STR_LOCAL_LEN(strvar)
#endif
#else
#if defined(SIDL_F90_STR_STRUCT_STR_LEN) || defined(SIDL_F90_STR_STRUCT_LEN_STR)
struct SIDL_F90_String_t;
typdef struct SIDL_F90_String_t *SIDL_F90_String;
#define SIDL_F90_STR(strvar) ((*strvar).str)
#define SIDL_F90_STR_LOCAL(strvar) struct SIDL_F90_String_t strvar = \
  { NULL, 0 }
#define SIDL_F90_STR_LOCAL_ARG(strvar) &strvar
#define SIDL_F90_STR_LOCAL_STR(strvar) ((strvar).str)
#define SIDL_F90_STR_LOCAL_LEN(strvar) ((strvar).len)
#define SIDL_F90_STR_LEN(strvar) ((*strvar).len)
#define SIDL_F90_STR_NEAR_LEN_DECL(strvar)
#define SIDL_F90_STR_FAR_LEN_DECL(strvar)
#else
#ifdef SIDL_F90_STR_STRUCT_STR_LEN
struct SIDL_F90_String_t {
  char *str;
  int  len;
};
#else
struct SIDL_F90_String_t {
  int  len;
  char *str;
};
#endif
#endif
#endif
#define SIDL_F90_STR_COPY(localvar,argvar,minsize) \
  sidl_copy_ior_str(&(SIDL_F90_STR_LOCAL_STR(localvar)), \
                    &(SIDL_F90_STR_LOCAL_LEN(localvar)), \
                    (argvar), (minsize))

#ifndef SIDL_F90_STR_MINSIZE
#define SIDL_F90_STR_MINSIZE 512
#endif

typedef int SIDL_F90_Bool;
#ifndef SIDL_F90_TRUE
#define SIDL_F90_TRUE 1
#endif
#ifndef SIDL_F90_FALSE 
#define SIDL_F90_FALSE 0
#endif

#endif /* not FORTRAN90_DISABLED */


#ifdef __cplusplus
extern "C" { /*}*/
#endif


/*
 * Convert a FORTRAN string to a nul terminated C string.  The
 * returned pointer is a dynamically allocated copy of fstr.  Any
 * trailing space characters in fstr are left out of the copy.
 */
char *
sidl_copy_fortran_str(const char *fstr,
                      int         flen);

/*
 * Convert a C string into a FORTRAN string.  If the C string is
 * shorter than the fstr, fstr will be padded with space characters.
 * If the C string is longer than fstr, the first flen characters of
 * the C string are copied.
 */
void
sidl_copy_c_str(char       *fstr,
                int         flen,
                const char *cstr);

/*
 * Provide a block of memory and the length for a FORTRAN string.
 * This will dynamically allocate a string with max(strlen(iorstr),
 * minsize)+1 characters.  The last character in the buffer is
 * initialized to '\0'.
 * 
 * The pointer to the allocated space is stored in *newfstr, and the length
 * of the space is stored in *newflen.  If iorstr is not null, the contents
 * are copied into the allocated space.  Any uninitialized characters in
 * *newfstr are initialized to space characters.
 *
 * The intent is that the FORTRAN client will only use characters 0 through
 * minsize - 1 leaving the nul character untouched.
 */
void
sidl_copy_ior_str(char      **newfstr, 
                  int        *newflen,
                  const char *iorstr,
                  const int   minsize);

/*
 * Remove trailing space characters.  This
 * routine will store a nul character in index 0 to
 * buflen (inclusive) (i.e. a full string causes
 * the assignment buffer[buflen] = '\0'). This means
 * that buffer is actually buflen + 1 bytes long.
 * It returns buffer.
 */
char *
sidl_trim_trailing_space(char *buffer,
                         int   buflen);

#ifdef __cplusplus
}
#endif
#endif /* included_sidlfortran_h */
