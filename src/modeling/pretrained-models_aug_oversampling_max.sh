#!/bin/sh

geo=(rdcrop)
ker=(gnoise)
clim=(0.14)
blim=(0.2)
oversam_n=(1 2 3)
oversam_max=(2000 4000 8000)
for g in "${geo[@]}"; do
    for k in "${ker[@]}"; do
        for c in "${clim[@]}"; do
            for b in "${blim[@]}"; do
                for osn in "${oversam_n[@]}"; do
                for osm in "${oversam_max[@]}"; do
                python3 src/modeling/pretrained-models_aug.py generator.auglist.geo=$g \
                    generator.auglist.ker=$k \
                    generator.auglist.clim=$c \
                    generator.auglist.blim=$b \
                    generator.oversampling_n=$osn \
                    generator.oversampling_max=$osm
                sleep 30
                done
                done
            done
        done
    done
done
