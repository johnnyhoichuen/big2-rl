array=()

# filter out characters in file name
# filename=zc_adsf_qwer132467_xcvasdfrqw
# echo ${filename//[^0-9]/}   # ==> 132467

header="slurm-"

# store all valid values to array
for FILE in *; do
    # filter, -s = silent
    if grep -s FILE "$header"; then
        continue
    fi

    # do echo $FILE
    # echo ${FILE//[^0-9]/}
    num=${FILE//[^0-9]/}
    array+=($num)
done

# check if correct
# echo "checking array"
# for ITEM in ${array[@]}; do
#     echo $ITEM
# done

# convert array to number
# echo ${array[1]}
# int(){ expr ${array[1]} - ${array[0]}}

# testing with subtraction
# d1="2"
# d2="1"
# let d=d1-d2
# echo $d

# showing
# expr ${array[1]} - ${array[0]}

# $((array[1]-array[0])) here is an expression
a=$((array[1]-array[0]))
# echo $a

# echo "finding max"
max="0"
for ITEM in ${array[@]}; do
    # echo $ITEM
    if (($ITEM>max)); then
        max=$ITEM
    fi
done

# echo $max
# echo slurm-${max}

filename=slurm-${max}.out

echo "-------------------------"
echo Opening $filename
echo "-------------------------"
echo -e ".\n.\n.\n\n"

cat $filename
