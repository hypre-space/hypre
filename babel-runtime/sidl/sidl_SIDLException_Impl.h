/*
 * File:          sidl_SIDLException_Impl.h
 * Symbol:        sidl.SIDLException-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.SIDLException
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
 * babel-version = 0.9.8
 */

#ifndef included_sidl_SIDLException_Impl_h
#define included_sidl_SIDLException_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif

#line 48 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidl.SIDLException._includes) */
/*
 * Would be better if the splicer id was more general than "includes" so
 * as to acknowledge forward declarations, defines, whatever that may be
 * specific to this file.
 */
struct sidl_SIDLException_Trace;
/* DO-NOT-DELETE splicer.end(sidl.SIDLException._includes) */
#line 57 "sidl_SIDLException_Impl.h"

/*
 * Private data for class sidl.SIDLException
 */

struct sidl_SIDLException__data {
#line 62 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidl.SIDLException._data) */
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
  struct sidl_SIDLException_Trace    *d_trace_head;
  struct sidl_SIDLException_Trace    *d_trace_tail;
  unsigned long int                   d_trace_length;
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException._data) */
#line 84 "sidl_SIDLException_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_SIDLException__data*
sidl_SIDLException__get_data(
  sidl_SIDLException);

extern void
sidl_SIDLException__set_data(
  sidl_SIDLException,
  struct sidl_SIDLException__data*);

extern void
impl_sidl_SIDLException__ctor(
  sidl_SIDLException);

extern void
impl_sidl_SIDLException__dtor(
  sidl_SIDLException);

/*
 * User-defined object methods
 */

extern char*
impl_sidl_SIDLException_getNote(
  sidl_SIDLException);

extern void
impl_sidl_SIDLException_setNote(
  sidl_SIDLException,
  const char*);

extern char*
impl_sidl_SIDLException_getTrace(
  sidl_SIDLException);

extern void
impl_sidl_SIDLException_addLine(
  sidl_SIDLException,
  const char*);

extern void
impl_sidl_SIDLException_add(
  sidl_SIDLException,
  const char*,
  int32_t,
  const char*);

#ifdef __cplusplus
}
#endif
#endif
