/*
 * File:          SIDL_BaseException_Impl.h
 * Symbol:        SIDL.BaseException-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.4
 * Release:       $Name: V1-9-0b $
 * Revision:      @(#) $Id: SIDL_BaseException_Impl.h,v 1.4 2003/04/07 21:44:31 painter Exp $
 * Description:   Server-side implementation for SIDL.BaseException
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
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
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.4
 */

#ifndef included_SIDL_BaseException_Impl_h
#define included_SIDL_BaseException_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseException_h
#include "SIDL_BaseException.h"
#endif

/* DO-NOT-DELETE splicer.begin(SIDL.BaseException._includes) */
/*
 * Would be better if the splicer id was more general than "includes" so
 * as to acknowledge forward declarations, defines, whatever that may be
 * specific to this file.
 */
struct SIDL_BaseException_Trace;
/* DO-NOT-DELETE splicer.end(SIDL.BaseException._includes) */

/*
 * Private data for class SIDL.BaseException
 */

struct SIDL_BaseException__data {
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseException._data) */
  /*
   * To Do...Give more consideration to the trade-offs that result from using
   * a single character string that grows when lines are added to the trace
   * versus the linked list approach given below.
   *
   * Advantages include:
   * o No need for the overhead associated with the list (i.e.,
   *   the structure, next, tail, and length).
   *
   * Disadvantages include:
   * o Cannot implement any other trace methods such as a line-by-line
   *   dump.
   */
  char                               *d_message;
  struct SIDL_BaseException_Trace    *d_trace_head;
  struct SIDL_BaseException_Trace    *d_trace_tail;
  unsigned long int                   d_trace_length;
  /* DO-NOT-DELETE splicer.end(SIDL.BaseException._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct SIDL_BaseException__data*
SIDL_BaseException__get_data(
  SIDL_BaseException);

extern void
SIDL_BaseException__set_data(
  SIDL_BaseException,
  struct SIDL_BaseException__data*);

extern void
impl_SIDL_BaseException__ctor(
  SIDL_BaseException);

extern void
impl_SIDL_BaseException__dtor(
  SIDL_BaseException);

/*
 * User-defined object methods
 */

extern char*
impl_SIDL_BaseException_getNote(
  SIDL_BaseException);

extern void
impl_SIDL_BaseException_setNote(
  SIDL_BaseException,
  const char*);

extern char*
impl_SIDL_BaseException_getTrace(
  SIDL_BaseException);

extern void
impl_SIDL_BaseException_addLine(
  SIDL_BaseException,
  const char*);

extern void
impl_SIDL_BaseException_add(
  SIDL_BaseException,
  const char*,
  int32_t,
  const char*);

#ifdef __cplusplus
}
#endif
#endif
