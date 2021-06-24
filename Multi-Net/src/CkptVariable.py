import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = '../model/TrainALL/model_multi_net.ckpt-0'

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)

var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print('Tensor name: ',key)
    # print(reader.get_tensor(key))
