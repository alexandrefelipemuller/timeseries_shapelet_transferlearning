grep -v @ $1.arff | grep "%" -v | tr ':' ',' | awk 'BEGIN{FS=OFS=","} {s=$NF; for (i=NF-1; i>=1; i--) s = s ","$i; print s }' | sed '/^$/d' > $1
function prepare () { cd $1; ~/unzip $1.zip; ../convert.sh $1_TRAIN; ../convert.sh $1_TEST; cd ..; }
function baixa() { wget http://www.timeseriesclassification.com/Downloads/$1.zip; mkdir $1; mv $1.zip $1; } 

