
#!/bin/bash
#from: https://github.com/MTG/discogs-vi-dataset/blob/main/utilities/shuffle_and_split.sh
#########################################################################################

if [ $# == 0 ]; then
    echo "Description: This script shuffles a JSONL file and splits it into N equal parts.
            Example: shuffle_and_split.sh jsonl_file N"
    echo "Usage: $0 param1 param2"
    echo "param1: jsonl_file (file path)"
    echo "param2: N (number of splits)"
    exit 0
fi

#########################################################################################

jsonl_file=$1
N=$2

#########################################################################################

# Shuffle the file
jsonl_shuf=$jsonl_file.random
echo 'Shuffling the file...'
shuf $jsonl_file  > $jsonl_shuf

# Count the number of lines
l=`wc -l $jsonl_shuf | cut -d' ' -f1`
echo 'Shuffled file contains' $l 'lines in total'
x=$(($l / $N))
x=$(($x + 1))
echo 'each split will contain' $x 'lines approximately'

# Split the file
echo 'Splitting the file...'
output_file=$jsonl_file.split.
split -l $x -d $jsonl_shuf $output_file
echo 'Output files have the following structure:'
echo $output_file

# Remove the shuffled file
rm $jsonl_shuf

echo 'Done!'