function prepare () { cd $1; ~/unzip $1.zip; ../convert.sh $1_TRAIN; ../convert.sh $1_TEST; cd ..; }
function baixa() { wget http://www.timeseriesclassification.com/Downloads/$1.zip; mkdir $1; mv $1.zip $1; } 

for serie in BeetleFly BirdChicken ECGFiveDays ECG200 CBF FaceFour FacesUCR Gun_Point ItalyPowerDemand Lightning7 Lightning2 MoteStrain OliveOil DiatomSizeReduction Coffee Symbols Beef SyntheticControl Trace TwoLeadECG; do
	baixa "$serie"
	prepare "$serie"
	cd "$serie"
	grep -v @ "$serie"_TRAIN.arff | grep "%" -v | tr ':' ',' | awk 'BEGIN{FS=OFS=","} {s=$NF; for (i=NF-1; i>=1; i--) s = s ","$i; print s }' | sed '/^$/d' > "$serie"_TRAIN
	grep -v @ "$serie"_TEST.arff | grep "%" -v | tr ':' ',' | awk 'BEGIN{FS=OFS=","} {s=$NF; for (i=NF-1; i>=1; i--) s = s ","$i; print s }' | sed '/^$/d' > "$serie"_TEST
done


