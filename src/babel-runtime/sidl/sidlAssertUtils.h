/*
 * File:        sidlAssertUtils.h
 * Revision:    @(#) $Revision: 1.5 $
 * Date:        $Date: 2006/08/29 22:29:48 $
 * Description: convenience C macros for managing SIDL Assertions
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

#ifndef included_sidlAssertUtils_h
#define included_sidlAssertUtils_h

/*
 * SIDL Assertion checking option descriptions.
 */
static const int s_CHECK_OFF             = 0;
static const int s_CHECK_TYPE_OFF        = 0;
static const int s_CHECK_PRECONDITIONS   = 1;
static const int s_CHECK_POSTCONDITIONS  = 2;
static const int s_CHECK_INVARIANTS      = 3;
static const int s_CHECK_PRE_POST_ONLY   = 4;
static const int s_CHECK_PRE_INV_ONLY    = 5;
static const int s_CHECK_POST_INV_ONLY   = 6;
static const int s_CHECK_ALL_TYPES       = 7;

static const int s_CHECK_FREQ_OFF        = 8;  /* for description only */
static const int s_CHECK_ALWAYS          = 9;
static const int s_CHECK_PERIODICALLY    = 10;
static const int s_CHECK_TIMING          = 11;
static const int s_CHECK_RANDOMLY        = 12;
static const int s_CHECK_ASSERTIONS      = 13; /* for description only! */

static const char* const s_CHECK_DESCRIPTION[] = {
  "no assertions", "preconditions", "postconditions", "invariants", 
  "pre- and post-conditions", "preconditions and invariants", 
  "postconditions and invariants", "all assertions", "",
  "always", "periodically", "timing", "randomly", "all frequencies",
};

/****************************************************************************
 * SIDL Assertion static support methods
 ****************************************************************************/
char* 
sidl_getCheckTypeDesc(int level);

char*
sidl_getCheckFrequencyDesc(int level);

char*
sidl_getCheckDescription(int level);

#endif /* included_sidlAssertUtils_h */
