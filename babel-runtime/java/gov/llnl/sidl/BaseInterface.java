//
// File:	BaseInterface.java
// Package:	gov.llnl.sidl
// Release:	$Name$
// Revision:	$Revision$
// Modified:	$Date$
// Description:	base interface for the sidl Java runtime system
//
// Copyright (c) 2000-2001, The Regents of the University of Calfornia.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the Components Team <components@llnl.gov>
// UCRL-CODE-2002-054
// All rights reserved.
// 
// This file is part of Babel. For more information, see
// http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
// for Our Notice and the LICENSE file for the GNU Lesser General Public
// License.
// 
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License (as published by
// the Free Software Foundation) version 2.1 dated February 1999.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// conditions of the GNU Lesser General Public License for more details.
// 
// You should have recieved a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

package gov.llnl.sidl;

/**
 * Interface <code>BaseInterface</code> is the base interface for all sidl
 * Java classes in the run-time system.  This class provides support for
 * obtaining the IOR pointer using the <code>_get_ior</code> method.
 */
public interface BaseInterface {
  /**
   * Return a long reference to the sidl IOR object.
   */
  abstract public long _get_ior();

  /**
   * Cast this object to the specified sidl name.  If the cast is invalid,
   * then return null.  If the cast is successful, then the returned object
   * can be cast to the proper Java type using a standard Java cast.
   */
  abstract public BaseInterface _cast2(String name);

  /**
   * The <code>addRef</code> method will be implemented by the sidl
   * base object class.
   */
  public abstract void addRef();
}
