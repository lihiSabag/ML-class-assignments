import math
import csv
from scipy.spatial import distance
import scipy.spatial as ss
point = [1, 0, 0, '?']
data1 = [1, 1, 1, 'a']
data2 = [1, 2, 0, 'b']

#1
print(f"The vector {data1[0]},{data1[1]},{data1[2]} has tag {data1[3]}")
print(f"The vector {data2[0]},{data2[1]},{data2[2]} has tag {data2[3]}\n")


#2
def euclideanDistance(instance1, instance2, length):
   distance = 0
   for x in range(length):
         #print ('x is ' , x)
         num1=float(instance1[x]) #type casting from the string they are saved as, to numbers.
         num2=float(instance2[x])
         distance += pow(num1-num2, 2)
   return math.sqrt(distance)

print("The distance between data1 to data 2 is: {}\n".format(euclideanDistance(data1, data2,3)))

#3
with open('myFile.csv', 'r') as myCsvfile:
    lines = csv.reader(myCsvfile)
    dataWithHeader = list(lines)

#put data in dataset without header line
dataset = dataWithHeader[1:]
print("The first vector : {}".format(dataset[0]))
print("The second vector : {}\n".format(dataset[1]))
res = euclideanDistance(dataset[0], dataset[1],len(dataset[0][:3]))
print("The distance between first vector to second vector is: {}\n".format(res))


class distClass:
    dist = -1 #distance of current point from test point
    tag = '-' #tag of current point

    def __init__(self, dist, tag):
        self.dist = dist
        self.tag = tag


#4
distObj = distClass(res, dataset[1][-1])
print(distObj.dist, distObj.tag)


def createDistList(dataset, instance):

    distList = []
    for data in dataset:
        d = euclideanDistance(instance, data, len(instance)-1)
        #print(f"i={i}, f={d} ' tag={dataset[i][-1]}")
        distList.append(distClass(d,data[-1]))
    # 6
    distList.sort(key=lambda x: x.dist)
    return distList

#5
createDistList(dataset,dataset[0])

#7
def most_common(lst):
    return max(set(lst), key=lst.count)

def myKNN(dataset,k,instance):
    n=[]
    list = createDistList(dataset,instance)

    for i in range(k):
        n.append(list[i].tag)
    print(n)
    tag =most_common(n)
    return tag






############ mytrain ###########
with open('myTrain.csv', 'r') as Csvfile:
    lines = csv.reader(Csvfile)
    dataWithHeader = list(lines)

#put data in dataset without header line
TrainDataset = dataWithHeader[1:]

# print('\n########### myFile ###########')
# print(f"tag for {dataset[0]} with k=1: ", myKNN( dataset, 1, dataset[0]))
# print(f"tag for {dataset[0]} with k=2: ", myKNN( dataset, 2, dataset[0]))
# print(f"tag for {dataset[0]} with k=3: ", myKNN( dataset, 3, dataset[0]))
# print(f"tag for {dataset[0]} with k=4: ", myKNN( dataset, 4, dataset[0]))
#
# print('\n########### mytrain ###########')
# print(f"tag for TrainDataset[0]  with k=1: ", myKNN( TrainDataset, 1, TrainDataset[0]))
# print(f"tag for TrainDataset[0] with k=2: ", myKNN( TrainDataset, 2, TrainDataset[0]))
# print(f"tag for TrainDataset[0] with k=3: ", myKNN( TrainDataset, 3, TrainDataset[0]))
# print(f"tag for TrainDataset[0] with k=4: ", myKNN( TrainDataset, 4, TrainDataset[0]))
# print(f"tag for TrainDataset[0] with k=5: ", myKNN( TrainDataset, 30, TrainDataset[0]))

#8
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def createMDistList(dataset, instance,distanceMetric):

    distList = []
    for data in dataset:
        d = distanceMetric(instance[:1], data[:1])
        distList.append(distClass(d,data[-1]))
    # 6
    distList.sort(key=lambda x: x.dist)
    return distList

def myKNN2(dataset,k,instance,distanceMetric):
    n=[]
    list = createMDistList(dataset,instance,distanceMetric)

    for i in range(k):
        n.append(list[i].tag)
    print(n)
    tag =most_common(n)
    return tag


print('\n########### mytrain with Manhattan distance ###########')
print(f"tag for TrainDataset[0]  with k=1: ", myKNN2( TrainDataset, 1, TrainDataset[0],manhattan))


