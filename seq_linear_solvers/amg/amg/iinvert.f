c     
      subroutine iinvert(a,n,nd)
      implicit real*8 (a-h,o-z)
      dimension a(nd,nd)
      dimension d(100)
c     
      if(n.gt.100) stop 'matrix too large in invert'
      if(n.gt.nd)  stop 'dimension problem in invert'
c     
c     foreward elimination
c     
      do 40 i=1,n-1
         do 30 j=i+1,n
            c=a(j,i)/a(i,i)
            a(j,i)=-c
            do 10 k=1,i-1
               a(j,k)=a(j,k)-c*a(i,k)
 10         continue
            do 20 k=i+1,n
               a(j,k)=a(j,k)-c*a(i,k)
 20         continue
 30      continue
 40   continue
c     
c     backward elimination
c     
      do 50 i=1,n-1
         a(n,i)=a(n,i)/a(n,n)
 50   continue
      a(n,n)=1.0/a(n,n)
c     
      do 90 i=n-1,1,-1
         c=1.0/a(i,i)
         do 60 j=1,i-1
            a(i,j)=c*a(i,j)
 60      continue
         do 70 j=i+1,n
            d(j)=c*a(i,j)
            a(i,j)=0.0
 70      continue
         a(i,i)=c
         do 80 j=i+1,n
            do 85 k=1,n
               a(i,k)=a(i,k)-d(j)*a(j,k)
 85         continue
 80      continue
 90   continue

      return
      end
