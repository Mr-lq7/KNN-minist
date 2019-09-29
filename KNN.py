import numpy as np
import os
import gzip #
import threading
from datetime import datetime
#60000张训练集图片
#10000张测试集图片

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

CURRENT_DIRECTORY = 'PY_PROJECT/minist/'

#为全局变量，统计匹配的个数
matchCount = [0, 0, 0, 0, 0]
#规定数据流读入类型，以及以大端模式读取
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype = dt)[0]

def extract_images(input_file):
    with gzip.open(input_file,'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MINIST image file:%s' %(magic, input_file))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
  #      print(magic, num_images, rows, cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype = np.uint8)
        data = data.reshape(num_images, rows * cols)
        zipf.close()
        return np.minimum(data, 1)#相当于归一化，可有可无

def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MINIST label file:%s' %(magic, input_file))
        num_items = _read32(zipf)    
        buf = zipf.read(num_items)
   #     print(magic, num_items)
        labels = np.frombuffer(buf, dtype = np.uint8)
        zipf.close()
        return labels

def Knn_Classify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0] #数据集的行数，即样本数量
    init_shape = newInput.shape[0]#图片为28*28,init_shape=28*28=784
    newInput = newInput.reshape(1,init_shape)
    diff = np.tile(newInput, (numSamples, 1)) - dataSet#扩张
#   print(diff.shape)
    squareDiff = diff ** 2 #(test_xi - train_xi)的平方
    squareDist = np.sum(squareDiff, axis = 1)#分别求和
    distance = squareDist ** 0.5
    sortedDistIndices = np.argsort(distance)#相当于外部排序，提取相应的索引

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex

def Test(k, num):
    print('step1:load data...')
    train_x = extract_images(CURRENT_DIRECTORY + TRAIN_IMAGES)
    train_y = extract_labels(CURRENT_DIRECTORY + TRAIN_LABELS)
    test_x = extract_images(CURRENT_DIRECTORY + TEST_IMAGES)
    test_y = extract_labels(CURRENT_DIRECTORY + TEST_LABELS)
    
    #----------------------------------------------
#    match = 0
    #将测试集切成四等分
    numTestSamples = test_x.shape[0] / 4 #10000
#    numTestSamples = test_x.shape[0] 
#    begin = datetime.now()
    for i in range(int((num - 1) * numTestSamples), int(num * numTestSamples)):
#    for i in range(int(numTestSamples)):
        predict = Knn_Classify(test_x[i], train_x, train_y, k)
        if predict == test_y[i]:
            matchCount[num] += 1
            #match += 1


    #---------------------------------------------
    
#    accuracy = float(match) / numTestSamples
#    end = datetime.now()
#    print('运行了%d秒' % ((end - begin).seconds))
#    print('K = %d The classify accuracy is: %.2f%%' % (k, accuracy * 100))
    

def Calculate_Accuracy(k):
    total = 0
    for i in range(1,5):
        total += matchCount[i]
    accuracy = float(total) / 10000
    print('K = %d The classify accuracy is: %.2f%%' % (k, accuracy * 100))

if __name__ == "__main__":

    for i in range(1,16):
        t1 = threading.Thread(target = Test, args = (i, 1))
        t2 = threading.Thread(target = Test, args = (i, 2))
        t3 = threading.Thread(target = Test, args = (i, 3))
        t4 = threading.Thread(target = Test, args = (i, 4))
        matchCount = [0,0,0,0,0]
        begin = datetime.now()
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        total = 0
        #阻塞线程，方便最后进行正确率统计
        t1.join()
        t2.join()
        t3.join()
        t4.join()

        end = datetime.now()
        #---------------------------------------------
        print('----------------------------------------------------------')
        print('运行了%d秒' % ((end - begin).seconds))
        Calculate_Accuracy(i)
        print('----------------------------------------------------------')
        print('')

    #测试k = 3的时候的准确率96.26%
    #Test(3,1)
    #print(os.getcwd())
    


