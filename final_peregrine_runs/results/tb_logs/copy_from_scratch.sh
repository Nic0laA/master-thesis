#!/bin/bash

scratch_dir="/scratch-shared/scur2012/final_peregrine_runs"

# for run in AttentionUNet UNet_original UNet_pruned0 UNet_pruned10 UNet_pruned5 ViT ViT_pretrained; do
#for run in UNet_NoReinit UNet_original_2 AttentionUNet_2 UNet_Softloss ViT_pretrained_2; do
#for run in UNet_original_half; do
for run in UNet_pruned5_2; do

    if [ ! -d ${run} ]; then
        mkdir -p ${run}
    fi

    # Copy bounds
    cp -f ${scratch_dir}/${run}/bounds*     ${run}

    # Loop over rounds and log versions
    for round in {1..8}; do
        for version in 0 1; do
            
            folder=${scratch_dir}/${run}/trainer_${run}_R${round}/${run}_R${round}/version_${version}
            if [ ! -d ${folder} ]; then
                continue
            fi
            if [ ! -d ${run}/R${round} ]; then
                mkdir -p ${run}/R${round}
            fi


            for file in `ls ${folder}/events*`; do

                name=`basename -a ${file}`
                cp ${file}  ${run}/R${round}/R${round}_v${version}_${name}



            done

        done
    done






done
