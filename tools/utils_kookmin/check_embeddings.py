# check_embeddings.py
# Reference : https://mxnet.apache.org/versions/1.0.0/tutorials/python/predict_image.html
import os
import cv2
import numpy as np
import mxnet as mx
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def img_load():
    base_path = "/mnt/hdd_10tb_2/suprema_hdd_10tb_2/data/face_jpg_data/NIST_20/nist_20_torch_mtcnn_RGB_reproduced/mtcnn_"
    
    data_container = np.zeros((40, 3, 112, 112))
    
    for i in range(40):
        file_path = base_path + str(i) + ".ppm"
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            raise Exception

        img = np.swapaxes(img, 0,2)
        img = np.swapaxes(img, 1,2)

        data_container[i,:,:,:] = img

    return data_container

def main():
    image_size = [112, 112]

    data_container = img_load()
    # data normalization
    # mean 1.0/127.5
    # mean = (127.5, 127.5, 127.5)

    ctx = mx.gpu()
    #ctx = mx.cpu()
    prefix = "/home/yhji/Desktop/suprema_frvt/kookmin_recognition_mxnet_test2/models/r100-arcface-emore/model"
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 1)

    all_layers = sym.get_internals()
    sym = all_layers["fc1_output"]
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (40, 3, 112, 112))])
    mod.set_params(arg_params, aux_params)

    for i in range(data_container.shape[0]):
        #embeddings = mod.predict(data_container)
        mod.forward(Batch([mx.nd.array(data_container)]))

    net_out = mod.get_outputs()
    embeddings = net_out[0].asnumpy()

    #dist = np.square(result_1-result_2).sum()

    file_path = "/home/yhji/Documents/frvt/mxnet_model_test/mxnet_embeddings.txt"
    #embeddings_np = embeddings.asnumpy()
    #np.savetxt(file_path, embeddings_np)
    print("Done")

if __name__=="__main__":
    main()