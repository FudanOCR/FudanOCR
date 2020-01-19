GPU=2
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_nips /home/cjy/syn90_train_9000000data_lmdb	 \
	--valroot /home/cjy/ic15_test_lmdb \
	--workers 4 \
	--batchSize 64 \
	--niter 10000 \
	--lr 1 \
	--cuda \
	--experiment output_9000000/ \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder \
        --MORAN ./output_9000000/39000_0.5712.pth       
