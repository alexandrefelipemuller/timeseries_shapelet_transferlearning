for serie in BeetleFly BirdChicken ECGFiveDays ECG200 CBF FaceFour FacesUCR Gun_Point ItalyPowerDemand Lightning7 Lightning2 MoteStrain OliveOil DiatomSizeReduction Coffee Symbols Beef SyntheticControl Trace TwoLeadECG; do

mkdir "$serie"_shapelet
mkdir "$serie"_no_shapelet
python3 shapelets.py "$serie"

cp "$serie"/"$serie"_TEST "$serie"_shapelet/
cp "$serie"/"$serie"_TEST "$serie"_no_shapelet/
mv "$serie"_result_series_shapelet.txt "$serie"_shapelet/"$serie"_shapelet_TRAIN
mv "$serie"_result_series_no_shapelet.txt "$serie"_shapelet/"$serie"_no_shapelet_TRAIN

done
