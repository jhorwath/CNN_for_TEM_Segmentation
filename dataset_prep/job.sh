readarray a < file_list_short.txt
b=("${a[@]:0:10000}")

parallel --bar 'python3 BNL_Data_Seg_0404.py {1}' ::: ${b[@]}
