for big in 0 1; do for mt in inception inception_imagenet resnet; do for ds in CUB200 NAB BIRDSNAP; do OUTPUT_PREFIX=.results_ft_BIG${big} BIG=$big DATASET=$ds MODEL_TYPE=$mt BATCH_SIZE=24 ./train.sh ; done ; done ; done

