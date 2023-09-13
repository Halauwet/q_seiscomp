#!/bin/bash

#  backup seiscomp archive ke HDD
# ====================================

tahun="2068"
sc_archive="/home/sysop/seiscomp/var/lib/archive"
backup_dir="/media/sysop/LaCie/PGR_Backup/archive/tes/tes"

if [ ! -d "$backup_dir" ]; then
    mkdir -p "$backup_dir"
    echo "Directory '$backup_dir' created."
fi

rsync -avz $sc_archive/$tahun $backup_dir

