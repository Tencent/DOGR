# Introduction
LLaVA-NeXT的NPU版本代码，且使用来自于InternVL-8B的internvit-300M作为vision encoder

# Install
sh scripts/install.sh

# Pre-training
Pretraining with LLaVA-558K

`
sh scripts/train/pretrain_internvit.sh
`

# SFT

## SFT with our-own dataset, corresponding to LLaVA-NeXT stage 1.5

`
sh scripts/train/finetune_internvit_stage1_5.sh
`

## SFT with LLaVA-665K

`
sh scripts/train/finetune_internvit.sh
`

## SFT with LLaVA-OneVision Data

TBD