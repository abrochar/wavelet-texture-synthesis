#export CUDA_VISIBLE_DEVICES=1

ITL=2000
SIZE=256
NC=128
MODEL=rf_gray_nobias_nit"$ITL"_size"$SIZE"

#NAME=turb_zoom
#python synthesise0.py -name $NAME -f ./results_gray/"$NAME"_"$MODEL".mat -s $SIZE -n $ITL -c $NC

#NAME=cerise512
#python synthesise0.py -name $NAME -f ./results_gray/"$NAME"_"$MODEL".mat -s $SIZE -n $ITL -c $NC

NAME=cerise
python synthesise0.py -name $NAME -f ./results_gray/"$NAME"_"$MODEL".mat -s $SIZE -n $ITL -c $NC

#NAME=gravel
#python synthesise0.py -name $NAME -f ./results_gray/"$NAME"_"$MODEL".mat -s $SIZE -n $ITL -c $NC

#NAME=bricks
#NAME=redpeppers
#python synthesise0.py -name $NAME -f ./results_gray/"$NAME"_"$MODEL".mat -s $SIZE -n $ITL -c $NC

#NAME=pebbles
#python synthesise0.py -name $NAME -f ./results_gray/"$NAME"_"$MODEL".mat -s $SIZE -n $ITL -c $NC
