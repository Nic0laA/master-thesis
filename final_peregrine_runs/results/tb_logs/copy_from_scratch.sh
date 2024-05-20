#!/bin/bash

scratch_dir="/scratch-shared/scur2012/final_peregrine_runs"

for run in AttentionUNet UNet_original UNet_pruned0 UNet_pruned10 UNet_pruned5 ViT ViT_pretrained; do

    if [ ! -d ${run} ]; then
        mkdir -p ${run}
    fi

    # Copy bounds
    cp -f ${scratch_dir}/${run}/bounds*     ${run}







done
