# train
python tools/train.py --config ./configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py --work-dir ./experiments/glip/swin-l

# test
python tools/test.py ./configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py ./checkpoints/glip/glip_l_mmdet-abfe026b.pth --show-dir result_imgs --work-dir ./experiments/glip

# demo
python demo/image_demo.py demo/demo.jpg configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py --weights checkpoints/glip/glip_l_mmdet-abfe026b.pth --texts "bench . car"
python demo/glip_demo.py D:/project/GLIP/image/apparel/dress configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py --weights checkpoints/glip/glip_l_mmdet-abfe026b.pth --texts "day . night . face . pants . skirt . dress . t-shirt . shirt . sweater . coat . suit . shoes . heels . sneakers . bag . wallet . backpack . suitcase . watch . necklace . bracelet . ring . earrings ." --custom-entities

