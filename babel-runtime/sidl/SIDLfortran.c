/*
 * File:        SIDLfortran.c
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name$
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Functions for FORTRAN interoperability
 *
 * Copyright (c) 2000-2001, The Regents of the University of Calfornia.
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

#include "SIDLfortran.h"
#include <string.h>
#include <stdlib.h>

/*
 * Make a copy of the FORTRAN string excluding any trailing space
 * characters. 
 */
char *
SIDL_copy_f77_str(const char * restrict fstr,
                  int                   flen)
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
SIDL_copy_c_str(char * restrict       fstr,
                int                   flen,
                const char * restrict cstr)
{
  if (fstr && (flen > 0)) {
    int clen = (cstr ? strlen(cstr) : 0);
    if (clen > 0) {
      memcpy(fstr, cstr, ((flen < clen) ? flen : clen));
    }
    if (clen < flen) {
      memset(fstr + clen, ' ', flen - clen);
    }
  }
} 
                
void
SIDL_copy_ior_str(char      **newfstr,
                  int        *newflen,
                  const char *iorstr,
                  const int   minsize)
{
  const int iorLen = (iorstr ? strlen(iorstr) : 0);
  const int newLen = ((iorLen > minsize) ? iorLen : minsize);
  char *newStr = (char*) malloc(newLen+1);
  if (newStr) {
    if (iorLen) {
      (void)memcpy(newStr, iorstr, iorLen);
    }
    if (iorLen < newLen) {
      (void)memset(newStr + iorLen, ' ', newLen - iorLen);
    }
    /* put a nul character after the area that FORTRAN 77 will use */
    newStr[newLen] = '\0';
    *newfstr = newStr;
    *newflen = newLen;
  }
  else {
    *newfstr = NULL;
    *newflen = 0;
  }
}

char *
SIDL_trim_trailing_space(char * restrict buffer,
                         int             buflen)
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
