#!/bin/bash

source /scratch/Goyalp/miniconda3/etc/profile.d/conda.sh

conda activate str_koopman_embs

python --version 

EPOCHS=4000
EPOCHS_DECODER=600


DIR="./Results"
 _DIR="./_Results"

 if [ -d "$DIR" ]; then
    echo "'$DIR' found and now copying files in '_$DIR' for backup, please wait ..."
    cp -R $DIR $_DIR
    echo "deleting '$DIR' ....."
   rm -r ./Results/
 fi

 echo "##################################################################"
 echo "############# Running Pendulum Example               ##############"
 echo "##################################################################"

 cd Pendulum/
 python Example_Pendulum.py --confi_model cubic --epochs $EPOCHS
 echo ""
 python Example_Pendulum.py --confi_model quad --epochs $EPOCHS
 echo ""
 python Example_Pendulum.py --confi_model linear --epochs $EPOCHS
 echo ""
 jupyter nbconvert --execute --to notebook --inplace pendulum_error_plots.ipynb
 cd ..
 pwd

 echo "##################################################################"
 echo "############# Running NO Example               ##############"
 echo "##################################################################"

 cd Nonlinear_oscillator/
 python Example_NO.py --confi_model cubic --epochs $EPOCHS
 echo ""
 python Example_NO.py --confi_model quad --epochs $EPOCHS
 echo ""
 python Example_NO.py --confi_model linear --epochs $EPOCHS
 echo ""
jupyter nbconvert --execute --to notebook --inplace nlo_error_plots.ipynb
 cd ..
 pwd

 echo "##################################################################"
 echo "############# Running LV Example               ##############"
 echo "##################################################################"
 cd Lotka_Volterra/
 python Example_LV.py --confi_model cubic --epochs $EPOCHS
 echo ""
 python Example_LV.py --confi_model quad --epochs $EPOCHS
 echo ""
 python Example_LV.py --confi_model linear --epochs $EPOCHS
 echo ""
 jupyter nbconvert --execute --to notebook --inplace lv_error_plots.ipynb
 cd ..

echo "##################################################################"
echo "############# Running NLS Example               ##############"
echo "##################################################################"

cd NLS/
python NLS_Decoder.py --confi_decoder cnn --epochs $EPOCHS_DECODER > ./../Results/NLS_results.txt
python NLS_Decoder.py --confi_decoder quad --epochs $EPOCHS_DECODER >> ./../Results/NLS_results.txt
echo ""
python Example_NLS.py --confi_model linear --epochs $EPOCHS >> ./../Results/NLS_results.txt 
python Example_NLS.py --confi_model quad --epochs $EPOCHS >> ./../Results/NLS_results.txt
python Example_NLS.py --confi_model cubic --epochs $EPOCHS >> ./../Results/NLS_results.txt
python Example_NLS.py --confi_model linear_opinf --epochs $EPOCHS >> ./../Results/NLS_results.txt
echo ""
echo ""
cd ..


echo "##################################################################"
echo "############# Running Wave Example               ##############"
echo "##################################################################"

cd Wave/
python Wave_Decoder.py --confi_decoder cnn --epochs $EPOCHS_DECODER > ./../Results/Wave_results.txt
python Wave_Decoder.py --confi_decoder quad --epochs $EPOCHS_DECODER >> ./../Results/Wave_results.txt
echo ""
python Example_Wave.py --confi_model linear --epochs $EPOCHS >> ./../Results/Wave_results.txt
python Example_Wave.py --confi_model quad --epochs $EPOCHS >> ./../Results/Wave_results.txt
python Example_Wave.py --confi_model cubic --epochs $EPOCHS >> ./../Results/Wave_results.txt
python Example_Wave.py --confi_model linear_opinf --epochs $EPOCHS >> ./../Results/Wave_results.txt
echo ""
cd ..

bash transfer_figures_for_paper.sh



