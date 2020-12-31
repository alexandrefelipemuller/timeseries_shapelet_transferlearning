cat Trace_1_init_from_zero.txt | grep -o "Epoch.*" | sed 's/: val_accuracy did not improve from /,/g' | sed 's/: val_accuracy improved from /,/g' | sed 's/ to .*//g' | grep -o "[0-9]*,[0-9.]*"
