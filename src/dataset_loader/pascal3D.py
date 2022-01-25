import scipy.io
import os
import numpy as np
import cv2
from src.dataset_loader.datasetUtils import *
from src.module.function import *

# AEI_kmeans = np.array(
#     [[-0.37959412,  0.0896221,   0.05287072,  0.90319805,  0.99185765,  0.99559656],
#      [-0.83103055,  0.10626562,  0.0299934 ,  0.51789511,  0.9870735 ,  0.99758095],
#      [ 0.96297726,  0.68287062,  0.09528844, -0.21831189,  0.72485644,  0.99080621],
#      [ 0.53015532,  0.10014143,  0.02805288, -0.83132568,  0.99192195,  0.99745991],
#      [ 0.86861489,  0.10909378, -0.01550593,  0.45770347,  0.98740085,  0.99779991],
#      [-0.93655641,  0.08003212,  0.00114281, -0.29706842,  0.98999588,  0.99767647],
#      [ 0.02219816,  0.03370241,  0.00365841, -0.98923496,  0.9974772 ,  0.99869783],
#      [ 0.92599029,  0.07402331,  0.01549187, -0.33420168,  0.99187946,  0.99774758],
#      [-0.52019853,  0.10331311, -0.03536063, -0.8353981 ,  0.98985149,  0.99392136],
#      [ 0.45297786,  0.10298805, -0.06113805,  0.8753308 ,  0.98987731,  0.99512509]]
# )

AEI_kmeans = np.array(
    [
     [-0.47621441,  0.12350617,  0.0664845  , 0.87744119 , 0.9890067  , 0.9958762 ],
     [-0.79701111,  0.0897479 ,  0.01711335 , 0.59947227 , 0.99189595 , 0.99781593],
     [ 0.24718982,  0.03466764, -0.04019557 , 0.96675895 , 0.99782538 , 0.99749265],
     [-0.92900045,  0.12796586, -0.02959894 ,-0.36313565 , 0.98995881 , 0.998478],
     [-0.99093385,  0.30963827,  0.04399168 ,-0.03849395 , 0.94132017 , 0.99682202],
     [ 0.63991822,  0.08676728,  0.01266549, -0.76207316 , 0.99382095,  0.99825609],
     [-0.97041808, -0.04349611,  0.05179524, -0.2217277 ,  0.99625264,  0.99575635],
     [-0.75931431,  0.34058989,  0.08872104,  0.64092271 , 0.93278592,  0.99194674],
     [ 0.36390551,  0.10402673,  0.04212833, -0.92760847 , 0.99118484,  0.9969696 ],
     [-0.02068825,  0.04477927,  0.00249808,  0.99825918 , 0.99533083,  0.99785909],
     [-0.93060167,  0.09966637,  0.02097796,  0.35872446 , 0.99066755,  0.99873714],
     [ 0.559026  ,  0.1759681 , -0.07896671,  0.82671968,  0.98174557,  0.99525479],
     [-0.98538106, -0.04002435,  0.01965351,  0.13296903,  0.99573206,  0.99765323],
     [ 0.98356559,  0.00115694, -0.00883092, -0.14943953,  0.99380634,  0.99798697],
     [-0.63086791,  0.09089955,  0.04917653,  0.77313566,  0.99302347,  0.9975129 ],
     [ 0.96297726,  0.68287062,  0.09528844 ,-0.21831189,  0.72485644,  0.99080621],
     [ 0.89273093,  0.11288946,  0.02832545, -0.43880685,  0.99064152 , 0.99760898],
     [-0.36066788,  0.23448267,  0.3728395 ,  0.92279967,  0.95020208 , 0.91946087],
     [ 0.37875795,  0.19846965, -0.14137089,  0.92281866,  0.97781774 , 0.98612405],
     [ 0.43960992,  0.05180215 ,-0.03794313,  0.89593427,  0.99690567,  0.99759366],
     [ 0.95803985,  0.08486974 ,-0.00948285,  0.27001179,  0.98916025,  0.99752139],
     [-0.3050887 ,  0.06243963 , 0.05041205,  0.95028344,  0.99524137,  0.99639665],
     [-0.74702989,  0.11269824 ,-0.01525002, -0.65815991,  0.99057759,  0.9981728 ],
     [-0.29650311,  0.05262828 ,-0.0096012 , -0.95147564,  0.99610075,  0.99613081],
     [-0.51872658,  0.27368644 ,-0.43666126, -0.8388962 ,  0.94688394,  0.89374456],
     [-0.5108259 ,  0.11407817 ,-0.03612973, -0.85536679,  0.98887529,  0.99624544],
     [ 0.68169805,  0.07487039 ,-0.02669289,  0.72808247,  0.99466217,  0.99823521],
     [ 0.84553889,  0.14025878 ,-0.02059248,  0.52720822,  0.98423765,  0.99785371],
     [ 0.65659539,  0.52835238 ,-0.2444696,  0.74462741 , 0.84188589,  0.94337074],
     [ 0.01983669,  0.02214991,  0.00109092, -0.99691492 , 0.99825084,  0.99899499]
    ]
)

class dataLoaderSingleObject(object):
    def __init__(self,
                 Pascal3DDataPath=None,
                 trainOrVal='train',
                 ):
        self.dataStart = 0
        self.dataLength = 0
        self.epoch = 0

        self._Pascal3DDataPath = Pascal3DDataPath
        self._trainOrVal = trainOrVal
        self._isTrain = True
        if self._trainOrVal == 'train':
            self._isTrain = True
        elif self._trainOrVal == 'val':
            self._isTrain = False
        else:
            print('choose \'train\' or \'val\'')
            return
        self._dataPathList = []
        self._CAD3DShapes = {}

        print('set pascal3d dataset...')

        self._getTrainList()
        self._loadDataPath()
        self._load3DShapes()
        self._dataPathShuffle()

    def _getTrainList(self):
        print('set train or val list...')
        self._Pascal3DTrainList = []
        datasetList = os.listdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/'))
        for datasetName in datasetList:
            if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/', datasetName)):
                txtFileList = os.listdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/',datasetName))
                for txtFileName in txtFileList:
                    className = txtFileName.split('.')[0].split('_')[0]
                    trainval = txtFileName.split('.')[0].split('_')[-1]
                    if trainval==self._trainOrVal:
                        with open(os.path.join(self._Pascal3DDataPath, 'Image_sets/',datasetName,txtFileName)) as txtFilePointer:
                            dataPointList = txtFilePointer.readlines()
                            for i, dataPoint in enumerate(dataPointList):
                                if datasetName == 'pascal':
                                    dp = dataPoint.split('\n')[0].split(' ')[0]
                                    isTrue = int(dataPoint.split('\n')[0].split(' ')[-1])
                                    if int(isTrue)==1:
                                        self._Pascal3DTrainList.append(dp)
                                else:
                                    dp = dataPoint.split('\n')[0].split(' ')[0]
                                    self._Pascal3DTrainList.append(dp)
        print('done!')

    def _loadDataPath(self):
        print('load datapoint path...')
        datasetList = os.listdir(os.path.join(self._Pascal3DDataPath, 'training_data'))
        for datasetName in datasetList:
            if datasetName == 'imagenet' or datasetName == 'pascal':
                dataPointList = os.listdir(os.path.join(self._Pascal3DDataPath, 'training_data', datasetName))
                for dataPointName in dataPointList:
                    if dataPointName in self._Pascal3DTrainList:
                        if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'training_data', datasetName, dataPointName)):
                            dataPointPath = os.path.join(self._Pascal3DDataPath, 'training_data', datasetName, dataPointName)
                            self._dataPathList.append(dataPointPath)
        self._dataPathList = np.array(self._dataPathList)
        self.dataLength = len(self._dataPathList)
        print('done!')

    def _dataPathShuffle(self):
        # print('')
        # print('data path shuffle...')
        self.dataStart = 0
        np.random.shuffle(self._dataPathList)
        self.dataLength = len(self._dataPathList)
        # print('done! : ' + str(self.dataLength))

    def _load3DShapes(self):
        print('load 3d shapes for pascal3d...')
        self._classIndex = {}
        self._CAD3DShapes = {}
        classList = os.listdir(os.path.join(self._Pascal3DDataPath, 'CAD'))
        classList.sort(key=natural_keys)
        classNum = 0
        for className in classList:
            if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'CAD', className)):
                if self._CAD3DShapes.get(className) is None:
                    self._classIndex[className] = classNum
                    self._CAD3DShapes[className] = []
                    classNum += 1
                CADModelList = os.listdir(os.path.join(self._Pascal3DDataPath, 'CAD', className))
                CADModelList.sort(key=natural_keys)
                for CADModel in CADModelList:
                    if CADModel.split(".")[-1] == 'npy':
                        shape = np.load(os.path.join(self._Pascal3DDataPath, 'CAD', className, CADModel)).reshape((64, 64, 64, 1))
                        self._CAD3DShapes[className].append(shape)
                self._CAD3DShapes[className] = np.array(self._CAD3DShapes[className])
                self._CAD3DShapes[className] = np.where(self._CAD3DShapes[className] > 0, 1.0, 0.0)
        self._classNum = classNum
        print('done!')

    def getKmeansAEI(self, k=None, max_iter=1000):
        AEI = []
        i = 0
        for dataPath in self._dataPathList:
            print(i,len(self._dataPathList))
            i += 1
            objFolderList = os.listdir(dataPath)
            objSelectedList = []
            for objFolder in objFolderList:
                if os.path.isdir(os.path.join(dataPath, objFolder)):
                    objInfoTXT = os.path.join(dataPath, objFolder, 'objInfo.txt')
                    with open(objInfoTXT) as objInfoPointer:
                        objInfo = objInfoPointer.readline()
                    className = objInfo.split(' ')[0]
                    if self._CAD3DShapes[className] is not None:
                        objSelectedList.append(objInfo)
                    for objIndex, objSelected in enumerate(objSelectedList):
                        className,imagePath,CADModelPath,colMin,rowMin,colMax,rowMax,azimuth,elevation,inPlaneRot=objSelected.split(' ')
                        azimuth, elevation, inPlaneRot = float(azimuth) / 180.0 * np.pi, float(elevation) / 180.0 * np.pi, float(inPlaneRot) / 180.0 * np.pi
                        sin = np.sin([azimuth, elevation, inPlaneRot])
                        cos = np.cos([azimuth, elevation, inPlaneRot])
                        AEI.append([sin[0], sin[1], sin[2], cos[0], cos[1], cos[2]])
        AEI = np.array(AEI)
        kmeans = kmeans_cosine(X=AEI, k=k, max_iter=max_iter)
        centers, xtoc, dist = kmeans.kmeansSample()
        return centers, xtoc, dist

    def getNextBatch(self, batchSizeof3DShape=32, imageSize=None, augmentation=True):
        inputImages = []
        outputImages, classList, instList, EulerRadList = [], [], [], []
        item_num = 0
        while len(outputImages)==0:
            for dataPath in self._dataPathList[self.dataStart:]:
                objFolderList = os.listdir(dataPath)
                np.random.shuffle(objFolderList)
                # objFolderList.sort()
                objSelectedList = []
                for objFolder in objFolderList:
                    objFolder = objFolder.decode("utf-8")
                    if os.path.isdir(os.path.join(dataPath, objFolder)):
                        objInfoTXT = os.path.join(dataPath, objFolder, 'objInfo.txt')
                        with open(objInfoTXT) as objInfoPointer:
                            objInfo = objInfoPointer.readline()
                        className = objInfo.split(' ')[0]
                        # if className == 'car':
                        #     objSelectedList.append(objInfo)
                        if self._CAD3DShapes[className] is not None:
                            objSelectedList.append(objInfo)
                np.random.shuffle(objSelectedList)
                if len(objSelectedList) > 0:
                    if len(outputImages) >= batchSizeof3DShape and len(inputImages)>0:
                        break
                    try:
                        image2DPath = os.path.join(self._Pascal3DDataPath, objSelectedList[0].split(' ')[1])
                        inputImage = cv2.imread(image2DPath, cv2.IMREAD_COLOR)
                        row_org, col_org, _ = inputImage.shape
                        for objIndex, objSelected in enumerate(objSelectedList):
                            className,imagePath,CADModelPath,colMin,rowMin,colMax,rowMax,azimuth,elevation,inPlaneRot=objSelected.split(' ')
                            colMin, rowMin, colMax, rowMax, azimuth, elevation, inPlaneRot = float(colMin), float(rowMin), float(
                                colMax), float(rowMax), float(azimuth), float(elevation), float(inPlaneRot)
                            width, height = colMax - colMin, rowMax - rowMin
                            if augmentation:
                                border_ratio = np.random.rand() * 0.2
                            else:
                                border_ratio = 0.1
                            colMin, rowMin = np.max((0, colMin - width * border_ratio)), np.max(
                                (0, rowMin - height * border_ratio))
                            colMax, rowMax = np.min((col_org, colMax + width * border_ratio)), np.min(
                                (row_org, rowMax + height * border_ratio))
                            image = inputImage[int(rowMin):int(rowMax), int(colMin):int(colMax), :]

                            is_flip = False
                            if augmentation:
                                is_flip = np.random.rand() > 0.5
                            if is_flip:
                                image = cv2.flip(image, 1)
                                if np.random.rand() > 0.5:
                                    image = cv2.flip(image, 0)
                            # image augmentation
                            image, _, _, _, _, _ = imageRandomAugmentation(
                                inputImage=image, imageRowFinal=imageSize[0], imageColFinal=imageSize[1],
                                imAug=augmentation, imAugPr=0.5, padding=False,
                                randomTrans=augmentation, randomScale=augmentation,
                                transPr=0.5, scalePr=0.5, transRatioMax=0.2,
                                scaleRatioMin=0.8, scaleRatioMax=1.2
                            )
                            image = image / 255.

                            azimuth, elevation, inPlaneRot = float(azimuth) / 180.0 * np.pi, float(elevation) / 180.0 * np.pi, float(inPlaneRot) / 180.0 * np.pi
                            # for crossdataset
                            # inPlaneRot = 0.
                            if is_flip:
                                azimuth = -azimuth
                            EulerRad = np.array([azimuth, elevation, inPlaneRot])
                            classVector = np.zeros(self._classNum)
                            classVector[self._classIndex[className]] = 1
                            cadIndex = int(CADModelPath.split('/')[-1])
                            instVector = np.zeros(10)
                            instVector[cadIndex - 1] = 1
                            # car 3d shape
                            if (cadIndex - 1 < 0) or (cadIndex > len(self._CAD3DShapes)):
                                for i in range(1000):
                                    print('pascal ', cadIndex - 1)
                                    return
                            inst3DCAD = self._CAD3DShapes[className][cadIndex - 1]

                            inputImages.append(image)
                            outputImages.append(inst3DCAD)
                            classList.append(classVector)
                            instList.append(instVector)
                            EulerRadList.append(EulerRad)
                            item_num += 1
                            if item_num >= batchSizeof3DShape:
                                break
                    except:
                        pass
                self.dataStart += 1
                if self.dataStart >= self.dataLength:
                    self.epoch += 1
                    self._dataPathShuffle()
                    break
        inputImages = np.array(inputImages).astype('float32')
        outputImages = np.array(outputImages).astype('float32')
        classList = np.array(classList).astype('float32')
        instList = np.array(instList).astype('float32')
        EulerRadList = np.array(EulerRadList).astype('float32')

        return instList, classList, np.sin(EulerRadList), np.cos(EulerRadList), inputImages, outputImages

# average image size of pascal3D : (508.54, 404.47)
class dataLoader(object):
    def __init__(self,
                 imageSize=(640,480),
                 gridSize=(20,15),
                 predNumPerGrid=5,
                 Pascal3DDataPath=None,
                 trainOrVal='train',
                 ):
        self.dataStart = 0
        self.dataLength = 0
        self.epoch = 0

        self._imageSize = imageSize
        self._gridSize = gridSize
        self._predNumPerGrid = predNumPerGrid
        self._Pascal3DDataPath = Pascal3DDataPath
        self._trainOrVal = trainOrVal
        self._isTrain = True
        if self._trainOrVal == 'train':
            self._isTrain = True
        elif self._trainOrVal == 'val':
            self._isTrain = False
        else:
            print('choose \'train\' or \'val\'')
            return
        self._dataPathList = []
        self._CAD3DShapes = None

        print('set pascal3d dataset...')

        self._getTrainList()
        self._loadDataPath()
        self._load3DShapes()
        self._dataPathShuffle()

    def _getTrainList(self):
        print('set train or val list...')
        self._Pascal3DTrainList = []
        datasetList = os.listdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/'))
        for datasetName in datasetList:
            if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/', datasetName)):
                txtFileList = os.listdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/',datasetName))
                for txtFileName in txtFileList:
                    className = txtFileName.split('.')[0].split('_')[0]
                    trainval = txtFileName.split('.')[0].split('_')[-1]
                    if className=='car' and trainval==self._trainOrVal:
                        with open(os.path.join(self._Pascal3DDataPath, 'Image_sets/',datasetName,txtFileName)) as txtFilePointer:
                            dataPointList = txtFilePointer.readlines()
                            for i, dataPoint in enumerate(dataPointList):
                                if datasetName == 'pascal':
                                    dp = dataPoint.split('\n')[0].split(' ')[0]
                                    isTrue = int(dataPoint.split('\n')[0].split(' ')[-1])
                                    if int(isTrue)==1:
                                        self._Pascal3DTrainList.append(dp)
                                else:
                                    dp = dataPoint.split('\n')[0].split(' ')[0]
                                    self._Pascal3DTrainList.append(dp)
        print('done!')

    def _loadDataPath(self):
        print('load datapoint path...')
        datasetList = os.listdir(os.path.join(self._Pascal3DDataPath, 'training_data'))
        for datasetName in datasetList:
            if datasetName == 'imagenet' or datasetName == 'pascal':
                dataPointList = os.listdir(os.path.join(self._Pascal3DDataPath, 'training_data', datasetName))
                for dataPointName in dataPointList:
                    if dataPointName in self._Pascal3DTrainList:
                        if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'training_data', datasetName, dataPointName)):
                            dataPointPath = os.path.join(self._Pascal3DDataPath, 'training_data', datasetName, dataPointName)
                            self._dataPathList.append(dataPointPath)
        self._dataPathList = np.array(self._dataPathList)
        self.dataLength = len(self._dataPathList)
        print('done!')

    def _dataPathShuffle(self):
        print('')
        print('data path shuffle...')
        self.dataStart = 0
        np.random.shuffle(self._dataPathList)
        self.dataLength = len(self._dataPathList)
        print('done! : ' + str(self.dataLength))

    def _load3DShapes(self):
        print('load 3d shapes for pascal3d...')
        self._CAD3DShapes = []
        CADModelList = os.listdir(os.path.join(self._Pascal3DDataPath, 'CAD', 'car'))
        CADModelList.sort()
        for CADModel in CADModelList:
            if CADModel.split(".")[-1] == 'npy':
                shape = np.load(os.path.join(self._Pascal3DDataPath, 'CAD', 'car', CADModel)).reshape(64, 64, 64, 1)
                self._CAD3DShapes.append(shape)
        self._CAD3DShapes = np.array(self._CAD3DShapes)
        self._CAD3DShapes = np.where(self._CAD3DShapes>0, 1.0, 0.0)
        print('done!')

    def _getOffset(self, batchSize):
        offsetX = np.transpose(np.reshape(
            np.array([np.arange(self._gridSize[0])]*self._gridSize[1]*self._predNumPerGrid),
            (self._predNumPerGrid, self._gridSize[1], self._gridSize[0])), (1,2,0))
        offsetX = np.tile(np.reshape(offsetX, (1,self._gridSize[1],self._gridSize[0],self._predNumPerGrid)),[batchSize,1,1,1])
        offsetY = np.transpose(np.reshape(
            np.array([np.arange(self._gridSize[1])]*self._gridSize[0]*self._predNumPerGrid),
            (self._predNumPerGrid, self._gridSize[0], self._gridSize[1])), (2,1,0))
        offsetY = np.tile(np.reshape(offsetY, (1,self._gridSize[1],self._gridSize[0],self._predNumPerGrid)),[batchSize,1,1,1])
        return offsetX.astype('float32'), offsetY.astype('float32')

    def getNextBatch(self, batchSizeof3DShape=32, imageSize=None, gridSize=None):
        if imageSize!=None:
            self._imageSize = imageSize
        if gridSize!=None:
            self._gridSize = gridSize
        inputImages, bboxImages, objnessImages, eulerRadImages = [], [], [], []
        outputImages, instList = [], []
        while len(outputImages)==0:
            for dataPath in self._dataPathList[self.dataStart:]:
                objFolderList = os.listdir(dataPath)
                np.random.shuffle(objFolderList)
                # objFolderList.sort()
                objSelectedList = []
                for objFolder in objFolderList:
                    if os.path.isdir(os.path.join(dataPath, objFolder)):
                        objInfoTXT = os.path.join(dataPath, objFolder, 'objInfo.txt')
                        with open(objInfoTXT) as objInfoPointer:
                            objInfo = objInfoPointer.readline()
                        className = objInfo.split(' ')[0]
                        if className == 'car':
                            objSelectedList.append(objInfo)
                if len(objSelectedList) > 0:
                    if len(outputImages) + len(objSelectedList) > batchSizeof3DShape and len(inputImages)>0:
                        break
                    try:
                        image2DPath = os.path.join(self._Pascal3DDataPath, objSelectedList[0].split(' ')[1])
                        inputImage = cv2.imread(image2DPath, cv2.IMREAD_COLOR)

                        is_flip = np.random.rand() > 0.5
                        if is_flip:
                            inputImage = cv2.flip(inputImage, flipCode=1)

                        # imageRow, imageCol, channel = inputImage.shape
                        # # make this a square image
                        # imgRowColMax = np.max((imageRow, imageCol))
                        # heightBorderSize = (imgRowColMax - imageRow) / 2
                        # widthBorderSize = (imgRowColMax - imageCol) / 2
                        # inputImage = cv2.copyMakeBorder(
                        #     inputImage, top=heightBorderSize, bottom=heightBorderSize,
                        #     left=widthBorderSize, right=widthBorderSize, borderType=cv2.BORDER_CONSTANT,
                        #     value=[0, 0, 0])

                        imageRowOrg, imageColOrg, _ = inputImage.shape
                        inputImage, rowPad, dRow, colPad, dCol, scale = imageRandomAugmentation(
                            inputImage=inputImage, imageRowFinal=self._imageSize[1], imageColFinal=self._imageSize[0],
                            imAug=self._isTrain, imAugPr=0.5, padding=False,
                            randomTrans=self._isTrain, randomScale=self._isTrain,
                            transPr=0.5, scalePr=0.5, transRatioMax=0.2,
                            scaleRatioMin=0.8, scaleRatioMax=1.2
                        )
                        inputImage = inputImage/255.

                        imageRow = imageRowOrg + 2*rowPad
                        imageCol = imageColOrg + 2*colPad

                        objOrderingImage = -1 * np.ones([self._gridSize[1], self._gridSize[0], self._predNumPerGrid])
                        bboxImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 4])
                        objnessImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                        eulerRadImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
                        outputPerImage, instPerImage, EulerPerImage = [], [], []
                        itemIndex = 0
                        for objIndex, objSelected in enumerate(objSelectedList):
                            className,imagePath,CADModelPath,colMin,rowMin,colMax,rowMax,azimuth,elevation,inPlaneRot=objSelected.split(' ')
                            colMin, rowMin, colMax, rowMax, azimuth, elevation = float(colMin), float(rowMin), float(
                                colMax), float(rowMax), float(azimuth), float(elevation)
                            if is_flip:
                                colMin, colMax = imageColOrg - colMax, imageColOrg - colMin
                                azimuth, inPlaneRot = -azimuth, -inPlaneRot
                            # #border for square
                            # rowMin, rowMax = float(rowMin) + heightBorderSize, float(rowMax) + heightBorderSize
                            # colMin, colMax = float(colMin) + widthBorderSize, float(colMax) + widthBorderSize
                            # augmentation
                            colMin = (float(colMin) + colPad - imageCol/2) * scale + imageCol/2 + dCol
                            colMax = (float(colMax) + colPad - imageCol/2) * scale + imageCol/2 + dCol
                            rowMin = (float(rowMin) + rowPad - imageRow/2) * scale + imageRow/2 + dRow
                            rowMax = (float(rowMax) + rowPad - imageRow/2) * scale + imageRow/2 + dRow
                            # colMin = np.max((0.0, colMin))
                            # colMax = np.min((colMax, imageCol))
                            # rowMin = np.max((0.0, rowMin))
                            # rowMax = np.min((rowMax, imageRow))
                            azimuth,elevation,inPlaneRot = -float(azimuth)/180.0*np.pi,-float(elevation)/180.0*np.pi,float(inPlaneRot)/180.0*np.pi
                            cadIndex = int(CADModelPath.split('/')[-1])

                            rowCenterOnGrid = (rowMax+rowMin)/2.0*self._gridSize[1]/imageRow
                            colCenterOnGrid = (colMax+colMin)/2.0*self._gridSize[0]/imageCol
                            rowIndexOnGrid = int(rowCenterOnGrid)
                            colIndexOnGrid = int(colCenterOnGrid)
                            dx,dy = colCenterOnGrid - colIndexOnGrid, rowCenterOnGrid - rowIndexOnGrid
                            bboxHeight,bboxWidth = np.min((1.0, (rowMax-rowMin)/imageRow)),np.min((1.0, (colMax-colMin)/imageCol))
                            for predIndex in range(self._predNumPerGrid):
                                if (rowIndexOnGrid>=0 and rowIndexOnGrid<self._gridSize[1]) \
                                    and (colIndexOnGrid>=0 and colIndexOnGrid<self._gridSize[0]) \
                                    and bboxHeight > 0 and bboxWidth > 0 \
                                    and objnessImage[rowIndexOnGrid, colIndexOnGrid, predIndex] != 1:
                                    # objectness and bounding box
                                    objnessImage[rowIndexOnGrid,colIndexOnGrid,predIndex]=1
                                    bboxImage[rowIndexOnGrid,colIndexOnGrid,predIndex,0:4] = bboxHeight,bboxWidth,dx,dy
                                    eulerRadImage[rowIndexOnGrid,colIndexOnGrid,predIndex,:] = azimuth, elevation, inPlaneRot

                                    # car instance vector
                                    carInstVector = np.zeros(len(self._CAD3DShapes))
                                    carInstVector[cadIndex-1] = 1
                                    # car 3d shape
                                    if(cadIndex-1<0) or (cadIndex>len(self._CAD3DShapes)):
                                        for i in range(1000):
                                            print('pascal ', cadIndex-1)
                                            return
                                    car3DCAD = self._CAD3DShapes[cadIndex-1]
                                    # # Euler angle in rad
                                    # EulerRad = np.array([azimuth,elevation,inPlaneRot])

                                    # append items
                                    outputPerImage.append(car3DCAD)
                                    instPerImage.append(carInstVector)
                                    # EulerPerImage.append(EulerRad)
                                    # set item order
                                    objOrderingImage[rowIndexOnGrid, colIndexOnGrid, predIndex] = itemIndex
                                    itemIndex += 1
                                    break
                        if itemIndex > 0:
                            inputImages.append(inputImage)
                            bboxImages.append(bboxImage)
                            objnessImages.append(objnessImage)
                            eulerRadImages.append(eulerRadImage)

                            for gridRow in range(self._gridSize[1]):
                                for gridCol in range(self._gridSize[0]):
                                    for predIndex in range(self._predNumPerGrid):
                                        objOrder = int(objOrderingImage[gridRow, gridCol, predIndex])
                                        if objOrder>=0:
                                            outputImages.append(outputPerImage[objOrder])
                                            instList.append(instPerImage[objOrder])
                                            # EulerRadList.append(EulerPerImage[objOrder])
                    except:
                        pass
                self.dataStart += 1
                if self.dataStart >= self.dataLength:
                    self.epoch += 1
                    self._dataPathShuffle()
                    break
        inputImages = np.array(inputImages).astype('float32')
        objnessImages = np.array(objnessImages).astype('float32')
        bboxImages = np.array(bboxImages).astype('float32')
        eulerRadImages = np.array(eulerRadImages).astype('float32')
        outputImages = np.array(outputImages).astype('float32')
        instList = np.array(instList).astype('float32')
        # EulerRadList = np.array(EulerRadList).astype('float32')

        offsetX,offsetY = self._getOffset(batchSize=len(inputImages))

        # print inputImages.shape
        # print instList.shape
    #     return offsetX, offsetY, inputImages, objnessImages,\
    # bboxImages, np.sin(EulerRadList), np.cos(EulerRadList), \
    # outputImages, instList
        return offsetX, offsetY, inputImages, objnessImages, \
               bboxImages, np.sin(eulerRadImages), np.cos(eulerRadImages), \
               outputImages, instList


# data_loader_pascal = dataLoaderSingleObject(trainOrVal='val',
#                                              Pascal3DDataPath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/')
#
# center, xtoc, dist = data_loader_pascal.getKmeansAEI(k=30*12)
# print(center)
# print(np.median(dist))
# np.save('../../val.npy', center)




















