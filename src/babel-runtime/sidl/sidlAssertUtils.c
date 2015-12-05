/*
 * File:        sidlAssertUtils.c
 * Revision:    @(#) $Revision: 1.6 $
 * Date:        $Date: 2007/09/27 19:35:42 $
 * Description: code for managing SIDL Assertions
 *
 * Copyright (c) 2004, The Regents of the University of Calfornia.
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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "sidlAssertUtils.h"
#include "sidlAsserts.h"
#include "sidl_String.h"

char*
sidl_getCheckTypeDesc(int level) {
  int chk = s_CHECK_TYPE_OFF;
  switch (level & CHECK_ALL_TYPES) {
    case CHECK_ALL_TYPES:      chk = s_CHECK_ALL_TYPES;      break;
    case CHECK_POST_INV_ONLY:  chk = s_CHECK_POST_INV_ONLY;  break;
    case CHECK_PRE_INV_ONLY:   chk = s_CHECK_PRE_INV_ONLY;   break;
    case CHECK_INVARIANTS:     chk = s_CHECK_INVARIANTS;     break;
    case CHECK_PRE_POST_ONLY:  chk = s_CHECK_PRE_POST_ONLY;  break;
    case CHECK_POSTCONDITIONS: chk = s_CHECK_POSTCONDITIONS; break;
    case CHECK_PRECONDITIONS:  chk = s_CHECK_PRECONDITIONS;  break;
  }
  return sidl_String_strdup(s_CHECK_DESCRIPTION[chk]);
}

char* 
sidl_getCheckFrequencyDesc(int level) {
  int chk = s_CHECK_FREQ_OFF;
  switch (level & CHECK_ASSERTIONS) {
    case CHECK_ALWAYS:         chk = s_CHECK_ALWAYS;         break;
    case CHECK_PERIODICALLY:   chk = s_CHECK_PERIODICALLY;   break;
    case CHECK_TIMING:         chk = s_CHECK_TIMING;         break;
    case CHECK_RANDOMLY:       chk = s_CHECK_RANDOMLY;       break;
  }
  return sidl_String_strdup(s_CHECK_DESCRIPTION[chk]);
}

char*
sidl_getCheckDescription(int level) {
  char* type  = sidl_getCheckTypeDesc(level);
  char* freq  = sidl_getCheckFrequencyDesc(level);
  char* adapt = (level & CHECK_ADAPTIVELY) ? "adaptively, " : "";
  char* res   = (char*)malloc(strlen(type) + strlen(freq) + strlen(adapt) + 8);
  sprintf(res, "%s%s, %s", adapt, freq, type);
  free(type);
  free(freq);
  return res;
}
