# convert_mxnet2onnx.py
# code to converting mxnet model to onnx to use in opencv cpp api
# 210111 @Young-hoon Ji
# reference : https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/deploy/export/onnx.html
import os
import numpy as np
import mxnet
from mxnet.contrib import onnx as onnx_mxnet
import logging
import onnx
import cv2

logging.basicConfig(level=logging.INFO)

class mxnet2onnx:
    def __init__(self, model_dir):
        #print("init")
        self.model_dir = model_dir
        self.model = None

    def convert_mxnet2onnx(self, symbol_file, params_file, input_shape, output_file):
        print("Converting mxnet model to onnx")

        sym = self.model_dir + symbol_file
        params = self.model_dir + params_file
        model_input_shape = input_shape
        output_onnx_file = self.model_dir + output_file

        converted_model = onnx_mxnet.export_model(sym, params, [model_input_shape], np.float32, output_onnx_file)
        print("onnx model saved into", converted_model)


        return converted_model # return onnx model path

    def validation_onnx_converting(self, converted_model_path):
        print("Validate converted onnx file")

        model_proto = onnx.load_model(converted_model_path)
        print("onnx model loaded successfully from ", converted_model_path)

        # check if converted ONNX protobuf is valid
        onnx.checker.check_graph(model_proto.graph)
        print("No error in converting mxnet model to onnx")


        #return model_proto # return onnx model loaded

#     def dataloader(self, input_text_dir):
#         # Read Input
#         # Load image from file list
#         base_path = input_text_dir
#         enroll_src = open(base_path+"enroll.txt", 'r')
#         verif_src = open(base_path+"verif.txt", 'r')

#         data_enroll = enroll_src.readlines()
#         data_verif = verif_src.readlines()

#         #img_root = "/mnt/hdd_10tb_1/suprema/data/frvt_testset/aligned/"
#         img_root = "/mnt/hdd_10tb_2/suprema_hdd_10tb_2/data/verification/frvt_testset_aligned_python_test/aligned/"

#         #img_cv_1 = np.zeros([len(data), 112, 112, 3])
#         #img_cv_2 = np.zeros([len(data), 112, 112, 3])
#         img_cv_enroll= []
#         img_cv_verif = []

#         labels_enroll = []
#         labels_verif = []

#         #img_list = np.zeros([len(data)*2, 112, 112, 3])

#         embedding_enroll = []
#         embedding_verif = []

#         gt_label = []

#         for i in range(len(data_enroll)):
#             # from here need to load 
#             enroll_line = data_enroll[i]
#             verif_line = data_verif[i]
            
#             # TODO extract file name and 
#             enroll_target = enroll_line.split()[1]
#             enroll_file_name = enroll_target.split("/")[-1]
#             enroll_label = enroll_file_name.split("-")[0]
            
#             enroll_img = cv2.imread(img_root + enroll_file_name)
#             img_cv_enroll.append(enroll_img)
#             labels_enroll.append(enroll_label)
                    
#             verif_target = verif_line.split()[1]
#             verif_file_name = verif_target.split("/")[-1]
#             verif_label = verif_file_name.split("-")[0]
            
#             verif_img = cv2.imread(img_root + verif_file_name)
#             img_cv_verif.append(verif_img)
#             labels_verif.append(verif_label)

#             if(enroll_label==verif_label):
#                 gt_label.append(0)
#             else:
#                 gt_label.append(1)

#         img_cv_enroll = np.array(img_cv_enroll)
#         img_cv_verif = np.array(img_cv_verif)

#         #similarity_score_onnx = []

#         # Convert to Blob
#         alpha_ = 1.0/127.5
#         mean_ = (127.5, 127.5, 127.5)
#         size_ = (112,112)

#         # input shape : (112,112,3)
#         blob_1 = cv2.dnn.blobFromImages(img_cv_enroll, alpha_, size_, mean_)
#         blob_2 = cv2.dnn.blobFromImages(img_cv_verif, alpha_, size_, mean_)

#         return blob_1, blob_2, gt_label

#     def verification(self, onnx_model_path, blob_1, blob_2, gt_label):
#         self.model = cv2.dnn.readNetFromONNX(onnx_model_path)
#         print("Testing Model loaded")

#         embedding_enroll = []
#         embedding_verif = []
#         similarity_score_onnx = []

#         # Forward
#         for i in range(blob_1.shape[0]):
#             input_blob_1 = np.expand_dims(blob_1[i], axis=0)
#             self.model.setInput(input_blob_1)
#             result_1 = self.model.forward()

#             input_blob_2 = np.expand_dims(blob_2[i], axis=0)
#             self.model.setInput(input_blob_2) 
#             result_2 = self.model.forward()

#             embedding_enroll.append(result_1)
#             embedding_verif.append(result_2)

#             dist = np.square(result_1-result_2).sum()
#             Similarity = 4.0 - dist
#             similarity_score_onnx.append(Similarity)

#         embedding_enroll = np.array(embedding_enroll)
#         embedding_verif = np.array(embedding_verif)

#         with open("./plot/centos/similarity_score_frvt_test_201106_onnx_centos.txt", "wt") as f:
#             for i in range(len(data_enroll)):
#                 f.writelines("{}\n".format(similarity_score_onnx[i]))




# def nist_loader(onnx_model_path):
#     #onnx_model_path = "./onnx_model/kookmin_frvt11_001.onnx"

#     print(cv2.__version__)
#     print("Onnx Model Path: ", onnx_model_path)

#     # Load Model
#     #model = onnx.load(onnx_model_path)
#     model = cv2.dnn.readNetFromONNX(onnx_model_path)
#     #onnx.checker.check_model(model)
#     print("Loaded Model checked")

#     # Select Backend & Target
#     #model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#     #model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
#     # Read Input
#     # Load image from file list
#     f_src = open("/home/yhji/Desktop/data/torch_mtcnn_RGB_BGR_check/nist_20_torch_mtcnn_RGB_reproduced/target_label.txt", 'r')    
#     data = f_src.readlines()

#     img_root = "/home/yhji/Desktop/data/torch_mtcnn_RGB_BGR_check/nist_20_torch_mtcnn_RGB_reproduced"

#     img_cv_1 = []
#     img_cv_2 = []

#     #img_list = np.zeros([len(data)*2, 112, 112, 3])
#     img_list = []

#     embedding_1 = []
#     embedding_2 = []

#     for i, image_name in enumerate(data):
#         data1 = image_name.split(' ')[0]
#         img_path_1 = img_root + '/' + data1
#         image1 = cv2.imread(img_path_1)
#         #img_cv_1[i] = image1
#         img_cv_1.append(image1)

#         data2 = image_name.split(' ')[1]
#         img_path_2 = img_root + '/' + data2
#         image2 = cv2.imread(img_path_2)
#         img_cv_2.append(image2)
        
#     img_cv_1 = np.array(img_cv_1)
#     img_cv_2 = np.array(img_cv_2)

#     similarity_score_onnx = []

#     # Convert to Blob
#     alpha_ = 1.0/127.5
#     mean_ = (127.5, 127.5, 127.5)
#     size_ = (112,112)

#     # input shape : (112,112,3)
#     blob_1 = cv2.dnn.blobFromImages(img_cv_1, alpha_, size_, mean_)
#     blob_2 = cv2.dnn.blobFromImages(img_cv_2, alpha_, size_, mean_)

#     # Forward
#     for i in range(blob_1.shape[0]):
#         input_blob_1 = np.expand_dims(blob_1[i], axis=0)
#         model.setInput(input_blob_1)
#         result_1 = model.forward()

#         input_blob_2 = np.expand_dims(blob_2[i], axis=0)
#         model.setInput(input_blob_2) 
#         result_2 = model.forward()

#         embedding_1.append(result_1)
#         embedding_2.append(result_2)

#         dist = np.square(result_1-result_2).sum()
#         Similarity = 4.0 - dist
#         similarity_score_onnx.append(Similarity)

#     embedding_1 = np.array(embedding_1)
#     embedding_2 = np.array(embedding_2)

    
#     # post-Process
#     target_label = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

#     #plot_name_onnx = "plot_NIST_onnx_tf_mtcnn_RGB_201116"
#     #plot_histogram(similarity_score_onnx, target_label, plot_name_onnx)


#     with open("./plot/centos/similarity_nist_onnx_torch_RGB_201119.txt", "w") as f:
#         for i in range(len(data)):
#             f.writelines("{}\n".format(similarity_score_onnx[i]))

# def frvt_loader(onnx_model_path, enroll_txt, verif_txt):
#     print("Onnx Model Path: ", onnx_model_path)

#     # Load Model
#     model = cv2.dnn.readNetFromONNX(onnx_model_path)
#     print("ONNX Model Loaded")

#     #base_path = "/mnt/hdd_10tb_1/suprema/data/frvt_testset/original_txt_input/"
#     base_path = "/home/yhji/Desktop/data/frvt_testset_4python/input_frtv_testset_aligned_4python/"
#     enroll_src = open(base_path+"enroll.txt", 'r')
#     verif_src = open(base_path+"verif.txt", 'r')

#     data_enroll = enroll_src.readlines()
#     data_verif = verif_src.readlines()


#     #img_root = "/mnt/hdd_10tb_1/suprema/data/frvt_testset/aligned/"
#     img_root = "/home/yhji/Desktop/data/frvt_testset_4python/frvt_testset_aligned_python_test/aligned/"

#     #img_cv_1 = np.zeros([len(data), 112, 112, 3])
#     #img_cv_2 = np.zeros([len(data), 112, 112, 3])
#     img_cv_enroll= []
#     img_cv_verif = []

#     labels_enroll = []
#     labels_verif = []

#     #img_list = np.zeros([len(data)*2, 112, 112, 3])

#     embedding_enroll = []
#     embedding_verif = []

#     gt_label = []

#     for i in range(len(data_enroll)):
#         # from here need to load 
#         enroll_line = data_enroll[i]
#         verif_line = data_verif[i]
        
#         # TODO extract file name and 
#         enroll_target = enroll_line.split()[1]
#         enroll_file_name = enroll_target.split("/")[-1]
#         enroll_label = enroll_file_name.split("-")[0]
        
#         enroll_img = cv2.imread(img_root + enroll_file_name)
#         img_cv_enroll.append(enroll_img)
#         labels_enroll.append(enroll_label)
                
#         verif_target = verif_line.split()[1]
#         verif_file_name = verif_target.split("/")[-1]
#         verif_label = verif_file_name.split("-")[0]
        
#         verif_img = cv2.imread(img_root + verif_file_name)
#         img_cv_verif.append(verif_img)
#         labels_verif.append(verif_label)

#         if(enroll_label==verif_label):
#             gt_label.append(0)
#         else:
#             gt_label.append(1)

#     img_cv_enroll = np.array(img_cv_enroll)
#     img_cv_verif = np.array(img_cv_verif)

#     similarity_score_onnx = []

#     # Convert to Blob
#     alpha_ = 1.0/127.5
#     mean_ = (127.5, 127.5, 127.5)
#     size_ = (112,112)

#     # input shape : (112,112,3)
#     blob_1 = cv2.dnn.blobFromImages(img_cv_enroll, alpha_, size_, mean_)
#     blob_2 = cv2.dnn.blobFromImages(img_cv_verif, alpha_, size_, mean_)

#     # Forward
#     for i in range(blob_1.shape[0]):
#         input_blob_1 = np.expand_dims(blob_1[i], axis=0)
#         model.setInput(input_blob_1)
#         result_1 = model.forward()

#         input_blob_2 = np.expand_dims(blob_2[i], axis=0)
#         model.setInput(input_blob_2) 
#         result_2 = model.forward()

#         embedding_enroll.append(result_1)
#         embedding_verif.append(result_2)

#         dist = np.square(result_1-result_2).sum()
#         Similarity = 4.0 - dist
#         similarity_score_onnx.append(Similarity)

#     embedding_enroll = np.array(embedding_enroll)
#     embedding_verif = np.array(embedding_verif)


#     """
#     # post-Process
#     target_label = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

#     plot_name_onnx = "plot_NIST_onnx"
#     plot_histogram(similarity_score_onnx, target_label, plot_name_onnx)
#     """

#     with open("./plot/centos/similarity_score_frvt_test_201106_onnx_centos.txt", "wt") as f:
#         for i in range(len(data_enroll)):
#             f.writelines("{}\n".format(similarity_score_onnx[i]))

#     """
#     with open("./plot/201105_ppm/label_frvt_test.txt", "wt") as f:
#         for i in range(len(data_enroll)):
#             f.writelines("{}\n".format(gt_label[i]))
#     """

# def plot_histogram(result_array, label, filename):
#     for i in range(20):
#         X = label[i]
#         Y = result_array[i]

#         # plot 입력
#         scatter = plt.scatter(X, Y, s=10, c="gray", label='A')
        
#     # 그래프의 타이틀과 x, y축 라벨링
#     plt.title(filename, pad=10)
#     plt.xlabel('Genuine     Imposter', labelpad=10)
#     plt.ylabel('Similarity', labelpad=10)

#     # 플롯 출력
#     #plt.savefig('./scatter_frvt_report_target_similarity_yhji.png')
#     plot_name = "./plot/" + filename + ".png"
#     plt.savefig(plot_name)
    

if __name__=="__main__":
    model_dir_path = "/mnt/hdd_10tb_2/suprema_hdd_10tb_2/trained_model/mxnet_model/r100-arcface-emore/"
    
    model_symbol_file = "model-symbol.json"
    model_params_file = "model-0001.params"
    model_input_shape = (1, 3, 112, 112) # (batch, input channel, width, height)
    model_output_file = "model-mxnet_210113.onnx"

    if os.path.exists(model_dir_path) is not True:
        print("Check dir path")
        raise Exception("model dir not in path")
    
    mxnet_converter = mxnet2onnx(model_dir_path)
    converted_model_path = mxnet_converter.convert_mxnet2onnx(model_symbol_file, model_params_file, model_input_shape, model_output_file)

    mxnet_converter.validation_onnx_converting(converted_model_path)
    
    print("Done")
