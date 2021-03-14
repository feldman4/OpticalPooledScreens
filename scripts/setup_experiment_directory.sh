#!/bin/bash

mkdir -p input

SCRIPT_PATH=$(dirname "$0")

cp "${SCRIPT_PATH}/../resources/input_files.xlsx" .
cp "${SCRIPT_PATH}/../resources/OpticalPooledScreens.smk" .
cp "${SCRIPT_PATH}/../resources/example_config_process.yaml" config.yaml