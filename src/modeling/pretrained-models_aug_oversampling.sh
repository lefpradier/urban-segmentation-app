#!/bin/sh

geo=(rdcrop)
ker=(mblur gnoise)
clim=(0 0.07 0.14 0.2)
blim=(0.2)
oversam_n=(1 2)

for g in "${geo[@]}"; do
    for k in "${ker[@]}"; do
        for c in "${clim[@]}"; do
            for b in "${blim[@]}"; do
                for osn in "${oversam_n[@]}"; do
                python3 src/modeling/pretrained-models_aug.py generator.auglist.geo=$g \
                    generator.auglist.ker=$k \
                    generator.auglist.clim=$c \
                    generator.auglist.blim=$b \
                    generator.oversampling_n=$osn
                sleep 30
                done
            done
        done
    done
done
