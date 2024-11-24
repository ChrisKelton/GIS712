#!/bin/bash
cwd=$(realpath $(dirname $0))
base_dir="$cwd/data"
clc_dir="$cwd/clc-data"
#
base_dirs="$base_dir/ROIs1158_spring $base_dir/ROIs1970_fall"
for base_dir_ in $base_dirs; do
  for fn in $(find $base_dir_ -maxdepth 1 -type d); do
    #echo "fn: $fn"
    rel_base_path=$(basename $fn)
    rel_base_path="$(basename $(dirname $fn))/$rel_base_path"
    if [ ! -d "$clc_dir/$rel_base_path" ]; then
      echo "$clc_dir/$rel_base_path does not exist! Making symlink."
      ln -s $fn $clc_dir/$rel_base_path
    fi
  done
done
