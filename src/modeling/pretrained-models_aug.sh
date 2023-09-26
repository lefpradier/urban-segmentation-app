#!/bin/sh

geo=(hflip rdcrop rotateb)
ker=(mblur gnoise)
clim=(0.2 0.4)
blim=(0.2 0.4)

for g in "${geo[@]}"; do
    for k in "${ker[@]}"; do
        for c in "${clim[@]}"; do
            for b in "${blim[@]}"; do
                python3 src/modeling/pretrained-models_aug.py generator.auglist.geo=$g \
                    generator.auglist.ker=$k \
                    generator.auglist.clim=$c \
                    generator.auglist.blim=$b
                sleep 30
            done
        done
    done
done
