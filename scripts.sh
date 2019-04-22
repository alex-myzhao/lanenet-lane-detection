# Test
python tools/test_lanenet.py --is_batch False --batch_size 1 --weights_path model/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000  --image_path data/carla_test_image/0.png

# Train
python tools/train_lanenet.py --net vgg --dataset_dir data/training_data_example/ --weights_path model/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000

# Visualize
tensorboard --logdir=/Users/myzhao/Projects/lanenet/lanenet-lane-detection/tboard/tusimple_lanenet/vgg

# Fix 'lanenet_model not found'
export PYTHONPATH="${PYTHONPATH}:/Users/myzhao/Projects/lanenet/lanenet-lane-detection-master/lanenet_model"
