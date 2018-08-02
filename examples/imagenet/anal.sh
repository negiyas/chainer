
awk '
/^CALLSTACK/ {
    command = $2
    memsize = $4
    maxstr = " "
    if (command == "MALLOC" ) {
        memtotal += memsize
        if (memtotal > memmax) {
             memmax = memtotal
             swapmax = swaptotal
             maxstr = "*"
        }
    } else if (command == "FREE" ) {
        memtotal -= memsize
    }
    printf("%2s%6s %14\047d %14\047d %14\047d # %s\n", maxstr, command, memsize,
           memtotal, swaptotal, $0)
}
/^SWAPOUT:/ {
    memsize = $4
    swaptotal += memsize
    printf("%7s %14\047d %14\047d %14\047d # %s\n", "SWAPOUT", memsize,
           memtotal, swaptotal, $0)
}
/^SWAPIN:/ {
    memsize = $4
    swaptotal -= memsize
    printf("%7s %14\047d %14\047d %14\047d # %s\n", "SWAPIN", memsize, memtotal,
           swaptotal, $0)
}
/^'\{\"hook\":\"malloc\",'/ {
    command = "MALLOC"
    memstr = substr($1, index($1, "mem_size") + 10)
    memsize = int(substr(memstr, 0, index(memstr, ",") - 1))
    memtotal += memsize
    maxstr = " "
    if (memtotal > memmax) {
         memmax = memtotal
         swapmax = swaptotal
         maxstr = "*"
    }
    printf("%1s%6s %14\047d %14\047d %14\047d # %s\n", maxstr, command, memsize,
           memtotal, swaptotal, $0)
}
/^'\{\"hook\":\"free\",'/ {
    command = "FREE"
    memstr = substr($1, index($1, "mem_size") + 10)
    memsize = int(substr(memstr, 0, index(memstr, ",") - 1))
    memtotal -= memsize
    maxstr = " "
    printf("%1s%6s %14\047d %14\047d %14\047d # %s\n", maxstr, command, memsize,
           memtotal, swaptotal, $0)
}

END {
    printf("%7s %14s %14\047d %14\047d\n", "MAX", "-", memmax, swapmax) \
    > "/dev/stderr"
}
' $*
