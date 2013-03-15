#vmd_rmsd.tcl
#   Calculates RMSD
#   Written by Karl Debiec on 13-03-06
#   Last updated 13-03-06

###################################################### FUNCTIONS #######################################################
proc rmsd {topology trajectory reference selection} {
    mol     new         $topology
    animate delete      beg 0 end 0 skip 0 0
    mol     addfile     $trajectory waitfor all top
    mol     new         $reference
    set     all_atoms   [atomselect 0 all]
    set     trj_sel     [atomselect 0 $selection]
    set     ref_sel     [atomselect 1 $selection]
    set     n_frames    [molinfo    0 get numframes]
    puts    "N_FRAMES $n_frames"
    for { set i 0 } { $i < $n_frames } { incr i } {
        $all_atoms      frame       $i
        $trj_sel        frame       $i
        set transmat    [measure fit $trj_sel $ref_sel]
        set Rxx         [lindex     [lindex $transmat 0] 0]
        set Rxy         [lindex     [lindex $transmat 1] 0]
        set Rxz         [lindex     [lindex $transmat 2] 0]
        set Ryx         [lindex     [lindex $transmat 0] 1]
        set Ryy         [lindex     [lindex $transmat 1] 1]
        set Ryz         [lindex     [lindex $transmat 2] 1]
        set Rzx         [lindex     [lindex $transmat 0] 2]
        set Rzy         [lindex     [lindex $transmat 1] 2]
        set Rzz         [lindex     [lindex $transmat 2] 2]
        $all_atoms      move        $transmat
        set rmsd        [measure rmsd $trj_sel $ref_sel]
        puts "ROTMAT    $Rxx $Rxy $Rxz $Ryx $Ryy $Ryz $Rzx $Rzy $Rzz"
        puts "RMSD      $rmsd"
    }
}

######################################################### MAIN #########################################################
set topology    [lindex $argv 0];
set trajectory  [lindex $argv 1];
set reference   [lindex $argv 2];
set selection   [lindex $argv 3];

rmsd $topology $trajectory $reference $selection
exit


