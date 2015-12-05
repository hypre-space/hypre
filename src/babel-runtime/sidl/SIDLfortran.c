/*
 * File:        sidlfortran.c
 * Revision:    @(#) $Revision: 1.8 $
 * Date:        $Date: 2007/09/27 19:35:48 $
 * Description: Functions for FORTRAN interoperability
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

#include "sidlfortran.h"
#include <string.h>
#include <stdlib.h>

/*
 * Make a copy of the FORTRAN string excluding any trailing space
 * characters. 
 */
char *
sidl_copy_fortran_str(const char * restrict fstr,
                      ptrdiff_t             flen)
{
  char *result;
  while ((flen > 0) && (' ' == fstr[flen-1])) {
    --flen;
  }
  if (flen < 0) {
    flen = 0;
  }
  if ((result = (char*) malloc(flen + 1))) {
    (void)memcpy(result, fstr, flen);
    result[flen] = '\0';
  }
  return result;
}

void
sidl_copy_c_str(char * restrict       fstr,
                size_t                flen,
                const char * restrict cstr)
{
  if (fstr && (flen > 0)) {
    size_t clen = (cstr ? strlen(cstr) : 0);
    if (clen > 0) {
      memcpy(fstr, cstr, ((flen < clen) ? flen : clen));
    }
    if (clen < flen) {
      memset(fstr + clen, ' ', flen - clen);
    }
  }
} 

static size_t
copy_ior_str(char       **newfstr,
             const char  *iorstr,
             const size_t minsize)
{
  const size_t iorLen = (iorstr ? strlen(iorstr) : 0);
  const size_t newLen = ((iorLen > minsize) ? iorLen : minsize);
  char *newStr = (char*) malloc(newLen+1);
  if (newStr) {
    if (iorLen) {
      (void)memcpy(newStr, iorstr, iorLen);
    }
    if (iorLen < newLen) {
      (void)memset(newStr + iorLen, ' ', newLen - iorLen);
    }
    /* put a null character after the area that FORTRAN will use */
    newStr[newLen] = '\0';
    *newfstr = newStr;
    return newLen;
  }
  else {
    *newfstr = NULL;
    return 0;
  }
}

#ifndef FORTRAN77_DISABLED                
void
sidl_f77_copy_ior_str(char               **newfstr,
                      SIDL_F77_String_Len *newflen,
                      const char          *iorstr,
                      const size_t         minsize)
{
  *newflen = (SIDL_F77_String_Len)copy_ior_str(newfstr, iorstr, minsize);
}
#endif

#ifndef FORTRAN90_DISABLED                
void
sidl_f90_copy_ior_str(char               **newfstr,
                      SIDL_F90_String_Len *newflen,
                      const char          *iorstr,
                      const size_t         minsize)
{
  *newflen = (SIDL_F90_String_Len)copy_ior_str(newfstr, iorstr, minsize);
}
#endif

char *
sidl_trim_trailing_space(char * restrict buffer,
                         ptrdiff_t       buflen)
{
  if (buflen >= 0 && buffer) {
    do {
      --buflen;
    }
    while ((buflen >= 0) && (buffer[buflen] == ' '));
    buffer[buflen+1] = '\0';
  }
  return buffer;
}
