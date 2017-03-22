# -*- coding:utf-8 -*-
import numpy as np

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dateSet, k):
    #得到有多少列
    n = np.shape(dateSet)[1]
    #创建k行，n列的矩阵
    centroids = np.mat(np.zeros((k, n)))
    #生成随机点，如果是二维的，那么随机点的列数必须和矩阵相同
    #生成出来的每一行可以看成是一个特征点，它的列数和矩阵相同，需要计算矩阵和这个随机点的距离
    for j in range(n):
        minJ = min(dateSet[:, j])
        rangeJ = float(max(dateSet[:,j]) - minJ)
        mins = minJ + rangeJ * np.random.rand(k, 1)
        centroids[:,j] = mins
    return centroids




def kMeans(dataSet, k, distMeas = distEclud ,createCent=randCent):
    #得到行数
    m = np.shape(dataSet)[0]
    #生成m行两列的矩阵
    clusterAssment = np.mat(np.zeros((m,2)))
    #得到中心点
    centroids = createCent(dataSet, k)
    #
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            #初始化最大距离为无穷远
            minDist = np.inf
            #初始化下标
            minIndex = -1
            for j in range(k):
                #将每一组行之间的距离记录下来
                distJI = distMeas(centroids[j, :], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist ** 2

            for cent in range(k):
                ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A) == cent][0]
                centroids[cent,:] = np.mean(ptsInClust, axis = 0)

        return centroids, clusterAssment




arr = np.array([[1,2,3,4,5],[2,3,4,5,1],[2,3,4,6,7]])

centroids, clusterAssment = kMeans(np.mat(arr), 3)

print 'cent',centroids

print 'data', clusterAssment
