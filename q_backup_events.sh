#!/bin/bash

#  backup seiscomp events ke HDD by eQ
# ====================================

tahun="2024"

backup_dir="/media/sysop/LaCie/PGR_Backup/event"

awal="$tahun-06-20 00:00:00"
akhir="$tahun-07-10 23:59:59"

if [ ! -d "$backup_dir" ]; then
    mkdir -p "$backup_dir"
    echo "Directory '$backup_dir' created."
fi


seiscomp exec scquery -d localhost/seiscomp eventlist_mysqlfin "$awal" "$akhir" >listevid.txt
hab=`wc -l listevid.txt|awk '{ print $1 }'`
let hab=(hab-1)
head -n$hab listevid.txt>listevid.txtok
jmf=`wc -l listevid.txtok|awk '{ print $1 }'`
while [ $jmf != 0 ]; do
         namaevent=`tail -n$jmf listevid.txtok|head -n1|awk '{ print $1 }'`
         seiscomp exec scxmldump -d mysql://sysop:sysop@localhost/seiscomp -E $namaevent -PAMfF -o $backup_dir/$namaevent.xml
         let jmf=(jmf-1)
         echo backup $backup_dir/$namaevent.xml ....
done

