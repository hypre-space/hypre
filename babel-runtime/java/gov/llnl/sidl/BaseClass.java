//
// File:	BaseClass.java
// Package:	gov.llnl.sidl
// Release:	$Name$
// Revision:	$Revision$
// Modified:	$Date$
// Description:	base class for the sidl Java runtime system
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

import gov.llnl.sidl.BaseInterface;
import java.lang.Exception;
import java.lang.reflect.Constructor;

/**
 * Class <code>BaseClass</code> is the base class for all sidl Java classes
 * in the run-time system.  This class provides support for dymaic loading,
 * Java JNI name registration, and reference counting.
 */
abstract public class BaseClass extends Exception implements BaseInterface {
  /*
   * Dynamically load the babel run-time library, which will contain
   * the necessary JNI dynamic symbols for this base class.
   */
  static {
      System.loadLibrary(System.getProperty("sidl.library.name", "sidl"));
   }

  /*
   * This data member is a reference to the sidl internal IOR structure.
   */
  protected long d_ior;

  /**
   * Register native functions for the sidl interface or class specified in
   * the argument.  This method must be called before any native method is
   * called on the Java object representing the sidl symbol.
   */
  public static native void _registerNatives(String sidl_symbol);

  /**
   * Construct a <code>BaseClass</code> object and initialize the IOR
   * reference to point to a valid sidl IOR structure.
   */
  protected BaseClass(long ior) {
    d_ior = ior;
  }

  /**
   * Retrieve a reference to the sidl IOR structure.
   */
  public final long _get_ior() {
    return d_ior;
  }

  /**
   * Cast this object to the specified sidl name.  If the cast is invalid,
   * then return null.  If the cast is successful, then the returned object
   * can be cast to the proper Java type using a standard Java cast.
   */
  public final BaseInterface _cast2(String name) {
    BaseInterface cast = null;

    /*
     * Cast this object to the specified type.  If the cast is valid, then
     * search for the matching Java type.
     */
    long ior = _cast_ior(name);
    if (ior != 0) {
      /*
       * Try to load either the Java interface wrapper or the class for this
       * sidl object.
       */
      Class java_class = null;
      try {
        java_class = Class.forName(name + "$Wrapper");
      } catch (Exception ex) {
	  //ignore this exception
        try {
          java_class = Class.forName(name);
        } catch (Exception ex2) {
	    ex.printStackTrace(System.err);     
        }
      }

      /*
       * If we found the class, then create a new instance using the sidl IOR.         */
      if (java_class != null) {
        Class[]  sigs = new Class[]  { Long.TYPE     };
        Object[] args = new Object[] { new Long(ior) };
        try {
          Constructor ctor = java_class.getConstructor(sigs);
          cast = (BaseInterface) ctor.newInstance(args);
          addRef();
        } catch (Exception ex) {
ex.printStackTrace(System.err);
          // ignore exception
        }
      }
    }

    return cast;
  }

  /**
   * Cast this object to the specified type and return the IOR pointer.
   */
  public final native long _cast_ior(String name);

  /**
   * On object destruction, call the native finalize method to reduce
   * the reference count on the IOR structure.
   */
  private final native void _finalize();

  /**
   * The finalizer of this method decreases the IOR reference count to
   * this Java object and then calls other finalizers in the chain.
   */
  protected void finalize() throws Throwable {
    _finalize();
    super.finalize();
  }
}
