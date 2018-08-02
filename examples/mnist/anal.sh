
awk '
/^CALLSTACK/ {
    maxstr = " "
    if ($2 == "MALLOC" ) {
        memtotal += $4
        if (memtotal > memmax) {
             memmax = memtotal
             swapmax = swaptotal
             maxstr = "*"
        }
    } else if ($2 == "FREE" ) {
        memtotal -= $4
    }
    printf("%1s%6s %14\047d %14\047d %14\047d # %s\n", maxstr, $2, $4,
           memtotal, swaptotal, $0)
}
/^SWAPOUT:/ {
    swaptotal += $4
    printf("%7s %14\047d %14\047d %14\047d # %s\n", "SWAPOUT", $4, memtotal,
           swaptotal, $0)
}
/^SWAPIN:/ {
    swaptotal -= $4
    printf("%7s %14\047d %14\047d %14\047d # %s\n", "SWAPIN", $4, memtotal,
           swaptotal, $0)
}
END {
    printf("%7s %14s %14\047d %14\047d\n", "MAX", "-", memmax, swapmax) \
    > "/dev/stderr"
}
' $*
