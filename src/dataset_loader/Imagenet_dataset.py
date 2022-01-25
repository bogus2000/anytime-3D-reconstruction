import os, cv2
import numpy as np
import sys
from src.dataset_loader.datasetUtils import *

class imagenetDataset(object):
    def __init__(self, dataPath, classNum=1000):
        self._dataPath = dataPath
        self._classNum = classNum
        self.epoch = 0
        self.dataStart = 0
        self.dataLength = 0
        self._dataPointPathList = None
        self._classIdxConverter = None
        self._imageSize = (480, 640)
        self._loadDataPointPath()
        self._dataShuffle()

    def setImageSize(self, size=(480, 640)):
        self._imageSize = (size[0],size[1])

    def _loadDataPointPath(self):
        print('load data point path...')
        self._dataPointPathList = []
        self._classIdxConverter = dict()
        trainPath = os.path.join(self._dataPath, 'train')
        classNameList = os.listdir(trainPath)
        classNameList.sort(key=natural_keys)
        classIdx = 0
        for className in classNameList:
            classPath = os.path.join(trainPath, className)
            if os.path.isdir(classPath):
                if className in self._classIdxConverter:
                    pass
                else:
                    self._classIdxConverter[className] = classIdx
                    classIdx += 1
                instNameList = os.listdir(classPath)
                instNameList.sort(key=natural_keys)
                for instName in instNameList:
                    instPath = os.path.join(classPath, instName)
                    self._dataPointPathList.append(instPath)
                sys.stdout.write('{:04d}/{:04d}\r'.format(classIdx, self._classNum))

            if classIdx == self._classNum:
                break
        self.dataLength = len(self._dataPointPathList)
        print('done!')

    def _dataShuffle(self):
        # 'data list shuffle...'
        self._dataStart = 0
        np.random.shuffle(self._dataPointPathList)
        # print 'done!'

    def getNextBatch(self, batchSize=32, imageSize=None):
        self._imageSize = imageSize
        if self.dataStart + batchSize >= self.dataLength:
            self.epoch += 1
            self.dataStart = 0
            self._dataShuffle()
        dataStart = self.dataStart
        dataEnd = dataStart + batchSize
        self.dataStart = self.dataStart + batchSize
        dataPathTemp = self._dataPointPathList[dataStart:dataEnd]
        inputImages = []
        # classIndexList = np.zeros((batchSize, self._classNum), dtype=np.float32)
        classIndexList = []
        for i in range(len(dataPathTemp)):
            try:
                imagePath = dataPathTemp[i]
                image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                image, rowPad, dRow, colPad, dCol, scale = imageRandomAugmentation(
                    inputImage=image, imageRowFinal=self._imageSize[0], imageColFinal=self._imageSize[1],
                    imAug=True, imAugPr=0.5,
                    randomTrans=True, randomScale=True,
                    transPr=0.5, scalePr=0.5, transRatioMax=0.2,
                    scaleRatioMin=0.8, scaleRatioMax=1.2
                )

                # if np.random.rand() < 0.5:
                #     image = datasetUtils.imgAug(image)

                # image = 2.0*image/255.0 - 1.0
                # print self._imageSize
                # image = cv2.resize(image, self._imageSize)
                # inputImages.append(image)
                inputImages.append(image.copy())

                className = imagePath.split("/")[-2]
                classIdx = self._classIdxConverter[className]
                classIndexVector = np.zeros(self._classNum)
                classIndexVector[classIdx] = 1
                # classIndexList[i,classIdx] = 1
                classIndexList.append(classIndexVector)

            except:
                pass

        batchData = {
            'input_images': np.array(inputImages).astype('float32'),
            'output_labels': np.array(classIndexList).astype('float32'),
        }
        return batchData