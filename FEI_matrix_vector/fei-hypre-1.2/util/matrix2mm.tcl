#!/usr/user3/baallan/usr/local/bin/tclsh8.0

set olddir [pwd]
set srcdir /usr/matrix_data
set destdir /usr/user3/baallan/matrix_data
cd $srcdir
set flist [glob *_matrix]
# let tcl handle each file in one pass and let cat
# handle gluing on a matrix market header
if {$srcdir == $destdir} {
	puts stderr "For paranoia reasons, srcdir and destdir must not be same"
	exit 1
}
foreach f $flist {
	if {![string compare $f testVBR_matrix]} {
		continue
	}
	set ifid [open $f r]
	set nr 0
	set nc 0
	set nnz 0
	set hfid [open $destdir/data.head w+]
	set bfid [open $destdir/data.body w+]
	while {![eof $ifid]} {
		if {![expr $nnz % 5000]} {
			puts -nonewline stderr "."
		}
		gets $ifid line
		# just echo comments into body.
		if {[string index $line 0] == "%"} {
			puts $bfid $line
			continue
		}
		switch [llength $line] {
			1 {
				set nr $line
			}
			3 {
				if {[lindex $line 1] > $nc} {
					set nc [lindex $line 1]
				}
				puts $bfid $line
				incr nnz
				continue
			}
			default {
				puts stderr "trash: $line"
			}
		}
	}
	close $ifid
	puts $hfid "%%MatrixMarket matrix coordinate real general"
	puts stdout "rows: $nr  columns: $nc  nonzeros: $nnz"
	puts $hfid "$nr $nc $nnz"
	close $hfid
	close $bfid
	
	file delete $destdir/$f
	puts stdout "Writing $destdir/$f"
	exec cat $destdir/data.head $destdir/data.body > $destdir/$f
	puts stdout "$destdir/$f done."
}
cd $olddir
