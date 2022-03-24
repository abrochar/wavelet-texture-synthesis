#export CUDA_VISIBLE_DEVICES=1

ITL=2000
SIZE=256
NC=128
MODEL=rf_color_nobias_nit"$ITL"_size"$SIZE"_nohist

#NAME=cerise
#NAME=bubbly
#python synthesise01.py -name $NAME -f ./results_color/"$NAME"_"$MODEL".mat -s $SIZE -n $ITL -c $NC

#NAME=cerise512
#NAME=flowers
NAME=pebbles
python synthesise01.py -name $NAME -f ./results_color/"$NAME"_"$MODEL".mat -s $SIZE -n $ITL -c $NC

#NAME=gravel
NAME=bricks
python synthesise01.py -name $NAME -f ./results_color/"$NAME"_"$MODEL".mat -s $SIZE -n $ITL -c $NC


