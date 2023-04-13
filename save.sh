#!/bin/bash
`
Script to upload zipped copy of code to server with todays date
as filename.
`

# Getting todays date as a variable
today="$(date +'%d_%m_%Y')"

# Compressing files + Ignoring hidden
zip -r "gislr".zip ./ -x "./wandb*" "*/.*" "*/_*"

# SCP to server for backup
# scp "$today".zip bartley@172.17.158.119:/home/bartley/gpu_test_backups/

# Deleting File locally
# rm "$today".zip