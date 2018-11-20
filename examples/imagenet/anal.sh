
awk '
function print_memory(line, kind, memsize, memfree) {
    maxstr = " "
    if (kind == "MALLOC") {
        memtotal += memsize
        if (memtotal > memmax) {
             memmax = memtotal
             swapmax = swaptotal
             maxstr = "*"
        }
    } else if (kind == "FREE") {
        memtotal -= memsize
    }
    if (maxmem == 0) {
       maxmem = memfree
    }
    printf("%1s%6s;%14\047d;%14\047d;%14\047d;%14\047d;# %s\n",
           maxstr, kind, memsize, memtotal, swaptotal, maxmem - memfree , line)
}
function print_swap(line, kind, swapsize) {
    if (kind == "SWAPOUT") {
        swaptotal += swapsize
    } else if (kind == "SWAPIN") {
        swaptotal -= swapsize
    }
    printf("%7s;%14\047d;%14\047d;%14\047d;;;# %s\n", kind, swapsize,
           memtotal, swaptotal, line)
}
function get_field(line, label) {
    idx = index($0, label) + length(label)
    start = substr($0, idx)
    len = index(start, ",")
    str = substr(start, 0, len - 1)
    return str
}
/^CALLSTACK/ {
    line = $0
    kind = $2
    memsize = $4
    memfree = $7
    print_memory(line, kind, memsize, memfree)
}
/^\{\"hook\":\"malloc\"/ {
    line = $0    
    kind = "MALLOC"
    memsize = get_field($0, "\"mem_size\":")
    memfree = 0 
    print_memory(line, kind, memsize, memfree)
}
/^\{\"hook\":\"free\"/ {
    line = $0    
    kind = "FREE"
    memsize = get_field($0, "\"mem_size\":")
    memfree = 0 
    print_memory(line, kind, memsize, memfree)
}
/^SWAPOUT:/ {
    line = $0
    kind = "SWAPOUT"
    swapsize = $4
    print_swap(line, kind, swapsize)
}
/^SWAPIN:/ {
    line = $0
    kind = "SWAPIN"
    swapsize = $4
    print_swap(line, kind, swapsize)
}
END {
    printf("%7s %14s %14\047d %14\047d\n", "TOTAL", "-", memtotal, swaptotal) \
    > "/dev/stderr"
    printf("%7s %14s %14\047d %14\047d\n", "MAX", "-", memmax, swapmax) \
    > "/dev/stderr"
}
' $*
