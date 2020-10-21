#!/bin/sh
DATA_PATH="/Users/cetinmehmet/Desktop/encryptedParq"
METRIC="Undefined"

while getopts s:p option
do
case "${option}"
in
s) SOURCE=${OPTARG};;
p) PERIOD=${OPTARG};;
esac
done

python3 /Users/cetinmehmet/Desktop/surfsara-tool/main/main.py --path=$DATA_PATH --source=$SOURCE  --period=FULL
