cat OliveOil_result_series_no_shapelet.txt_no_shapelet.txt | grep -o "Epoch.*\|val_accuracy...[.0-9]*" | tr -d '\n' | sed s/Epoch\ /\n/g |  sed s/.300val_accuracy:/,/
