#vmd_com_x.tcl
#   Outputs center of mass of atom selections across multiple segments
#   Written by Karl Debiec on 13-07-17
#   Last updated 13-07-18
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
package require pbctools
###################################################### FUNCTIONS #######################################################
proc com_x {topology trajectories selection_strings} {
    mol     new         $topology
    set     selections  [list]
    foreach selection_string $selection_strings {
        regsub -all {_} $selection_string " " selection_string
        lappend selections [atomselect top $selection_string]
    }

    set     i   0
    foreach trajectory $trajectories {
        mol     addfile     $trajectory waitfor all 0
        pbc     unwrap
        animate delete      beg 0 end 0 0
        set     n_frames    [molinfo    0 get numframes]
        puts    [format "SEGMENT %04d N_FRAMES %06d" $i $n_frames]
        for { set j 0 } { $j < $n_frames } { incr j } {
            set     k   0
            foreach selection $selections {
                $selection  frame   $j
                set         center  [measure center $selection]
                set         x       [lindex $center 0]
                set         y       [lindex $center 1]
                set         z       [lindex $center 2]
                puts        [format "FRAME %06d SELECTION %04d X %21.10f Y %21.10f Z %21.10f" $j $k $x $y $z]
                incr        k
            }
        }
        animate delete  beg 0 end [expr $n_frames - 2] 0
        incr    i
    }
}
######################################################### MAIN #########################################################
set trajectories      [list]
set selection_strings [list]
foreach {arg} $argv {
    if     { $arg    == "-topology"   } { set     status            "topology"   } \
    elseif { $arg    == "-trajectory" } { set     status            "trajectory" } \
    elseif { $arg    == "-selection"  } { set     status            "selection"  } \
    elseif { $status == "topology"    } { set     topology          $arg         } \
    elseif { $status == "trajectory"  } { lappend trajectories      $arg         } \
    elseif { $status == "selection"   } { lappend selection_strings $arg         }
}
com_x $topology $trajectories $selection_strings
exit
