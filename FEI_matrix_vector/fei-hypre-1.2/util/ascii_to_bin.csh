#!/bin/csh -f

# It is necessary to have first done "source setup" in the ISIS root dir to
# set environment variables, and to have made the conversion routine
# ascii_to_bin (done by issueing "make utils" from the ISIS root dir).

foreach prob (symmtest_10k)
	./convert/ascii_to_bin $MATRIX_DATA/$prob""_matrix $MATRIX_DATA_bin/$ISIS_ARCH/$prob""_matrix_bin $MATRIX_DATA/$prob""_rhs $MATRIX_DATA_bin/$ISIS_ARCH/$prob""_rhs_bin
end

echo "==> conversion from ascii to binary complete."

