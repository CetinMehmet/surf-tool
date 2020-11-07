#!/bin/sh
DATA_PATH="/Users/cetinmehmet/Desktop/encryptedParq"
METRIC="Undefined"

while getopts m:p:n: option
do
case "${option}"
in
m) METRIC=${OPTARG};;
p) PERIOD=${OPTARG};;
n) NODES=${OPTARG};;
esac
done

python3 /Users/cetinmehmet/Desktop/surfsara-tool/main/main.py --path=$DATA_PATH --metric=$METRIC  --period=$PERIOD --nodes=$NODES
