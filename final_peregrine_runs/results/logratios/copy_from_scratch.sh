#!/bin/bash

scratch_dir="/scratch-shared/scur2012/final_peregrine_runs"

# for run in AttentionUNet UNet_original UNet_pruned0 UNet_pruned10 UNet_pruned5 ViT ViT_pretrained; do
# for run in AttentionUNet AttentionUNet_2 UNet_original UNet_pruned0 UNet_pruned5 UNet_pruned10 UNet_NoReinit UNet_original_2 UNet_Softloss UNet_sampling ViT ViT_pretrained ViT_pretrained_2; do
# for run in UNet_original_half; do
for run in UNet_pruned5_2; do

    if [ ! -d ${run} ]; then
        mkdir -p ${run}
    fi

    # Copy logratios
    folder=${scratch_dir}/${run}/logratios_${run}
    cp ${folder}/*  ${run}

done

if [ ! -d Benchmark ]; then
    mkdir -p Benchmark
fi

cp /scratch-shared/scur2012/peregrine_data/bhardwaj2023/logratios_lowSNR/* Benchmark
