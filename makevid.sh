#!/bin/bash

set -e

cd /media/SURFlisa/runs/20170115T0905

if [ ! -d vid ]; then
    mkdir vid
fi


# "dm_peak_x" "dm_peak_y" "dm_peak_z" "halo_assignment"\
#"cygA_sampled_rho" "cygA_sampled_kT" \
#"cygNW_sampled_rho" "cygNW_sampled_kT"


#for f in `ls out/cygA*_donnert2014figure1_*`; do
#    snapnr=$(echo $f | cut -d'_' -f3 | cut -d'.' -f1)
#    mv -i $f "out/cygA_donnert2014figure1_${snapnr}.png"
#done
#exit 0


for base in "cygA_donnert2014figure1" "cygNW_donnert2014figure1" "xray_peakfind_00" \
            # "xray_peakfind_15" "xray_peakfind_30" "xray_peakfind_45" \
            # "xray_peakfind_60" "xray_peakfind_75"
do 
    ffmpeg -y -r 25 -i "out/${base}_%3d.png" -profile:v high444 -level 4.1 \
           -c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p -s "2000:2000" \
           -an "vid/${base}.mp4"
done
