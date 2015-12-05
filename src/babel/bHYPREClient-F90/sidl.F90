! 
! File:          sidl.F90
! Symbol:        sidl-v0.9.15
! Symbol Type:   package
! Babel Version: 1.0.0
! Release:       $Name: V2-4-0b $
! Revision:      @(#) $Id: sidl.F90,v 1.1 2007/02/06 01:23:05 painter Exp $
! Description:   Client-side module for sidl
! 
! Copyright (c) 2000-2002, The Regents of the University of California.
! Produced at the Lawrence Livermore National Laboratory.
! Written by the Components Team <components@llnl.gov>
! All rights reserved.
! 
! This file is part of Babel. For more information, see
! http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
! for Our Notice and the LICENSE file for the GNU Lesser General Public
! License.
! 
! This program is free software; you can redistribute it and/or modify it
! under the terms of the GNU Lesser General Public License (as published by
! the Free Software Foundation) version 2.1 dated February 1999.
! 
! This program is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
! conditions of the GNU Lesser General Public License for more details.
! 
! You should have recieved a copy of the GNU Lesser General Public License
! along with this program; if not, write to the Free Software Foundation,
! Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
! 
! WARNING: Automatically generated; changes will be lost
! 
! 


module sidl
  integer, parameter :: sidl_int=selected_int_kind(9)
  integer, parameter :: sidl_long=selected_int_kind(18)
  integer, parameter :: sidl_opaque=selected_int_kind(18)
  integer, parameter :: sidl_arrayptr=selected_int_kind(18)
  integer, parameter :: sidl_iorptr=selected_int_kind(18)
  integer, parameter :: sidl_enum=selected_int_kind(9)
  integer, parameter :: sidl_dcomplex=selected_real_kind(15,307)
  integer, parameter :: sidl_fcomplex=selected_real_kind(6,37)
  integer, parameter :: sidl_double=selected_real_kind(15,307)
  integer, parameter :: sidl_float=selected_real_kind(6,37)
end module sidl
