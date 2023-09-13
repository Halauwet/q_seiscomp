#!/bin/bash

#  backup seiscomp events ke HDD
# ====================================

backup_dir="/media/sysop/LaCie/PGR_Backup/event"
 
if [ ! -d "$backup_dir" ]; then
    mkdir -p "$backup_dir"
    echo "Directory '$backup_dir' created."
fi

awal="2015-01-01 00:00:00"
akhir="2022-12-31 23:59:59"

seiscomp exec scquery eventlist_mysqlfin $awal $akhir >listevid.txt
hab=`wc -l listevid.txt|awk '{ print $1 }'`
let hab=(hab-1)
head -n$hab listevid.txt>listevid.txtok
jmf=`wc -l listevid.txtok|awk '{ print $1 }'`
while [ $jmf != 0 ]; do
         namaevent=`tail -n$jmf listevid.txtok|head -n1|awk '{ print $1 }'`
         seiscomp exec scxmldump -d mysql://sysop:sysop@localhost/seiscomp3 -E $namaevent -PAMfF -o $backup_dir/$namaevent.xml
         let jmf=(jmf-1)
         echo backup $backup_dir/$namaevent.xml ....
done

