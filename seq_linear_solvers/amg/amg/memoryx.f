c
      subroutine memoryx(k,imin,imax,ia,ib,ipmn,ipmx)
c
c---------------------------------------------------------------------
c
c     set up amg components
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension ia (*)

      dimension ipmn(25),ipmx(25)

      dimension ib (*)
c
c---------------------------------------------------------------------
c
      write(6,999)
  999 format(/'  level ',i2,' storage requirements'/)
c
c     point storage
c
      write(6,1000)
 1000 format(/'  point information '/)
      iplo=ipmn(k)
      iphi=ipmx(k)
      write(6,1002) iplo,iphi
 1002 format('  xp     - ',2i10)
      write(6,1003) iplo,iphi
 1003 format('  yp     - ',2i10)
      write(6,1005) iplo,iphi+1
 1005 format('  iv     - ',2i10)
c
c     variable storage
c
      write(6,2000)
 2000 format(/'  variable information '/)
      ilo=imin(k)
      ihi=imax(k)
      write(6,2001) ilo,ihi
 2001 format('  u      - ',2i10)
      write(6,2002) ilo,ihi
 2002 format('  f      - ',2i10)
      write(6,2005) ilo,ihi
 2005 format('  icg    - ',2i10)
      write(6,2006) ilo,ihi
 2006 format('  ifg    - ',2i10)
      write(6,2007) ilo,ihi
 2007 format('  ip     - ',2i10)
      write(6,2006) ilo,ihi
 2009 format('  icgfix - ',2i10)
c
c     matrix storage
c
      write(6,3000)
 3000 format(/'  matrix information '/)
      write(6,2003) ilo,ihi+1
 2003 format('  ia     - ',2i10)
      jlo=ia(imin(k))
      jhi=ia(imax(k)+1)-1
      write(6,3001) jlo,jhi
 3001 format('  a      - ',2i10)
      write(6,3002) jlo,jhi+1
 3002 format('  ja     - ',2i10)
c
c     interpolation storage
c
      write(6,4000)
 4000 format(/'  interpolation information '/)
      write(6,2004) ilo,ihi+1
 2004 format('  ib     - ',2i10)
      jlo=ib(imin(k))
      jhi=ib(imax(k)+1)-1
      write(6,4001) jlo,jhi
 4001 format('  b      - ',2i10)
      write(6,4002) jlo,jhi+1
 4002 format('  jb     - ',2i10)
      return
      end


c
c=====================================================================
c
c     memory usage accounting routines
c
c=====================================================================
c
      subroutine memacct(array,nhigh,iperm)
c
c=====================================================================
c
c     this routine allows the user to store intermediate
c     memory usage for a number of arrays. This data is
c     stored in a common block, and can be accessed by
c     a sampling routine called by the user at reasonable
c     (often natural) intervals.
c
c     the usage can be designated as temporary or permanent.
c     temporary storage is work storage later discarded,
c     while permanent storage is retained until the run is
c     completed. permanent usage is nondecreasing. Temporary
c     usage is assumed to be higher than permanent usage.
c
c     Input:
c
c       array - character string containing array name
c       nhigh - amount of array used (any integer measure)
c       iperm - 0 = temporary, otherwise permanent
c
c     this routine stores the name for future reference. If
c     some particular order is desired, an initial call
c     with each of the array names (and nhigh=0) will ensure
c     a particular order for storing and reporting.
c
c     currently set for 20 names and 20 sampling times.
c
c---------------------------------------------------------------------
c
      integer nhigh
      character*(*) array
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias
c
c---------------------------------------------------------------------
c
c     initialize name counter
c
      call meminit(0)
c
c     search list for array name
c
      do 10 nn=1,nname
      if(array.eq.name(nn)) go to 30
10    continue
c
c     name not found. check for alias.
c
      do 20 nna=1,nalias
      if(array.eq.alias(nna)) then
        nn=ialias(nna)
        go to 30
      endif
20    continue
c
c     name not found. add to list and initialize usage to zero.
c
      if(nname.ge.maxname) then
        print *,' Too many names in MEMHIGH - name ignored...'
        return
      endif
c
      nname=nname+1
c
      name(nname)=array
      nn=nname
      memtemp(nn)=0
      memperm(nn)=0
      memmax (nn)=0
      memlim (nn)= 999999999
      if(nhigh.eq.0) return
c
c     update usage
c
30    if(iperm.eq.0) then
c
c     temporary use
c
        memtemp(nn)=nhigh
        if(nhigh.gt.memmax(nn)) memmax(nn)=nhigh
        if(nhigh.gt.memlim(nn)) write(6,
     *    '(/'' *** WARNING *** limit exceeded for '',a/)') name(nn)
      else
c
c     permanent use
c
        memperm(nn)=nhigh
        if(nhigh.gt.memtemp(nn)) memtemp(nn)=nhigh
        if(nhigh.gt.memmax(nn)) memmax(nn)=nhigh
        if(nhigh.gt.memlim(nn)) write(6,
     *    '(/'' *** WARNING *** limit exceeded for '',a/)') name(nn)
      endif
c
      return
      end
c
c=====================================================================
c
      subroutine meminit(irestart)
c
c=====================================================================
c
c     initializes name & sample counters
c     done if first call (init=0) or forced (irestart=0)
c
c---------------------------------------------------------------------
c
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias

      common /sample/ memtmps(maxname,maxsmpl),
     *                memprms(maxname,maxsmpl),
     *                memmaxs(maxname,maxsmpl),nsample

      data init /0/
      save
c
c---------------------------------------------------------------------
c
c     initialize name & sample counter
c
      if(init.eq.0.or.irestart.ne.0) then
        nname  =0
        nalias =0
        nsample=0
        init   =1
      endif
      return
      end
c
c
c=====================================================================
c
      subroutine msample
c
c=====================================================================
c
c     store current memory usage & increment sample counter
c
c---------------------------------------------------------------------
c
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias

      common /sample/ memtmps(maxname,maxsmpl),
     *                memprms(maxname,maxsmpl),
     *                memmaxs(maxname,maxsmpl),nsample
c
c---------------------------------------------------------------------
c
      if(nsample.ge.maxsmpl) then
        print *,' Limit reached in MSAMPLE - last data overwritten...'
      else
        nsample=nsample+1
      endif

      do 10 nn=1,nname
      memtmps(nn,nsample)=memtemp(nn)
      memprms(nn,nsample)=memperm(nn)
      memmaxs(nn,nsample)=memmax (nn)
10    continue
      return
      end
c
c=====================================================================
c
      subroutine memalias(newname,oldname)
c
c=====================================================================
c
c     allows accounting for "oldname" based on "newname"
c     in reports, only oldname is used.
c
c---------------------------------------------------------------------
c
cveh  integer nhigh
      character*(*) oldname,newname
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias
c
c---------------------------------------------------------------------
c
c     search alias list for newname - if it exists, return
c
      do 10 nna=1,nalias
      if(newname.eq.alias(nna)) return
10    continue
c
c     search list for oldname
c
      do 20 nn=1,nname
      if(oldname.eq.name(nn)) go to 30
20    continue
c
c     name not found. check for alias of alias.
c
30    do 40 nna=1,nalias
      if(oldname.eq.alias(nna)) then
        nn=ialias(nna)
        go to 50
      endif
40    continue
      stop ' old name does not exist'
c
c     add newname to alias list & reference oldname
c
50    nalias=nalias+1
c
      if(nalias.ge.maxname) then
        print *,' Too many names in MEMALIAS - name ignored...'
        return
      endif
c
      alias(nalias) =newname
      ialias(nalias)=nn
      return
      end
c
c=====================================================================
c
      subroutine mreport(ireport)
c
c=====================================================================
c
c     print report of memory usage
c
c       ireport = 1 - summary (max, permanent)
c       ireport = 2 - history + summary
c
c---------------------------------------------------------------------
c
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias

      common /sample/ memtmps(maxname,maxsmpl),
     *                memprms(maxname,maxsmpl),
     *                memmaxs(maxname,maxsmpl),nsample
c
c---------------------------------------------------------------------
c
      go to (100,200),ireport
      return
c
c     summary report
c
100   write(6,1000)
      do 110 nn=1,nname
      if(memlim(nn).eq.999999999) then
        write(6,1100) name(nn),memmax(nn),memperm(nn)
      else
        write(6,1200) name(nn),memmax(nn),memperm(nn),memlim(nn)
      endif
110   continue
      write(6,'(1x)')
c
1000  format(/'      SUMMARY OF ARRAY USAGE IN AMGS01 '/
     *        '      array        maximum        final      (limit)'/)
1100  format(6x,a7,5x,i8,5x,i8)
1200  format(6x,a7,5x,i8,5x,i8,5x,i8)
c
      return
c
c     history report
c
200   write(6,2000)
      do 250 nn=1,nname
      ns=1
      write(6,2100) name(nn),ns,memtmps(nn,ns),
     *              memprms(nn,ns),memmaxs(nn,ns)
      do 240 ns=2,nsample
      write(6,2200) ns,memtmps(nn,ns),
     *              memprms(nn,ns),memmaxs(nn,ns)
240   continue
      write(6,'(1x)')
250   continue
c
2000  format(/'      HISTORY OF TEMPORARY AND PERMANENT ARRAY USAGE'/
     *        '      array       k   temp use   perm use   max  use'/)
2100  format(6x,a7,4x,i2,3(3x,i8))
2200  format(17x,i2,3(3x,i8))
c
      return
      end
c
c=====================================================================
c
      subroutine memlimit(array,limit)
c
c=====================================================================
c
c     this routine allows the user to load array bounds
c     into the memory usage routines. Any further calls
c     will test against this limit and issue a warning if
c     it is exceeded. This can also be used to add the
c     array to the memory usage data.
c
c---------------------------------------------------------------------
c
      integer limit
      character*(*) array
      character*10  name,alias

      parameter(maxname=30,maxsmpl=30)

      common /memory/ name(maxname),alias(maxname),ialias(maxname),
     *                memtemp(maxname),memperm(maxname),
     *                memmax(maxname),memlim(maxname),
     *                nname,nalias
c
c---------------------------------------------------------------------
c
c     initialize name & sample counter
c
      call meminit(0)
c
c     search list for array name
c
      do 10 nn=1,nname
      if(array.eq.name(nn)) go to 30
10    continue
c
c     name not found. add to list and initialize usage to zero.
c
      if(nname.ge.maxname) then
        print *,' Too many names in MEMHIGH - name ignored...'
        return
      endif
c
      nname=nname+1
c
      name(nname)=array
      nn=nname
      memtemp(nn)=0
      memperm(nn)=0
      memmax (nn)=0
c
c     add limit
c
30    memlim (nn)=limit
c
      return
      end
