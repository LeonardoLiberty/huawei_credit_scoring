import math
import datetime
import sys
import numpy as np


class LR:
    def __init__(self, train_file_name, test_file_name, predict_result_file_name):
        self.train_file = train_file_name
        self.predict_file = test_file_name
        self.predict_result_file = predict_result_file_name
        self.max_iters = 10
        self.learning_rate = 0.005
        self.feats = []
        self.labels = []
        self.feats_test = []
        self.labels_predict = []
        self.param_num = 0
        self.weight = []

        self.loop_max = 800

    def loadDataSet(self, file_name, label_existed_flag):
        feats = []
        labels = []
        fr = open(file_name)
        lines = fr.readlines()
        for line in lines:
            temp = []
            allInfo = line.strip().split(',')
            dims = len(allInfo)
            sdims = dims//100 # this is for test with small data set
            if label_existed_flag == 1:
                for index in range(dims - 1):
                    temp.append(float(allInfo[index]))
                temp.append(1) # add the constant in the attributes
                feats.append(temp)
                labels.append(float(allInfo[dims - 1]))
            else:
                for index in range(dims): # +1 final
                    temp.append(float(allInfo[index]))
                temp.append(1)  # add the constant in the attributes
                feats.append(temp)
        fr.close()
        feats = np.array(feats)
        labels = np.array(labels)
        return feats, labels

    def loadTrainData(self):
        self.feats, self.labels = self.loadDataSet(self.train_file, 1)
        self.mean_training = np.mean(self.labels)
        '''
        Training set info
        '''
        print("======= training set info =========")

        print("label rate: " + str(self.mean_training))
        print("len features: "+ str(self.feats.shape))
        print("mean: ")
        # for i in range(self.feats.shape[1]):
        #     print(np.mean(self.feats[:,i]))

        print("======= finish training info ========")

    def loadTestData(self):
        self.feats_test, self.labels_predict = self.loadDataSet(
            self.predict_file, 0)

    def savePredictResult(self):
        print(self.labels_predict)
        f = open(self.predict_result_file, 'w')
        for i in range(len(self.labels_predict)):
            f.write(str(self.labels_predict[i])+"\n")
        f.close()

    def sigmod(self, x):
        return 1/(1+np.exp(-x))

    def printInfo(self):
        print(self.train_file)
        print(self.predict_file)
        print(self.predict_result_file)
        print(self.feats)
        print(self.labels)
        print(self.feats_test)
        print(self.labels_predict)

    def initParams(self):
        # self.weight = np.ones((self.param_num,), dtype=np.float)
        self.weight = np.random.rand(self.param_num,1)
        print("  ^^init weight^^  ")
        print(self.weight)

    def compute(self, recNum, param_num, feats, w):
        return self.sigmod(np.dot(feats, w))

    def error_rate(self, recNum, label, preval):
        return np.power(label - preval, 2).sum()

    def predict(self):
        self.loadTestData()
        preval = self.sigmod(np.dot( self.feats_test, self.weight))
        print("!! predict !!")
        preval = np.transpose(preval)[0]
        print(len(preval))
        print(preval)
        self.labels_predict = np.array([0 for i in range(len(preval))])
        for i in range(len(preval)):
            if preval[i] >= self.mean_training:
                self.labels_predict[i] = 1
        self.savePredictResult()

    def loss_function(self, x, y):
        return 0.5*(x-y)**2


    def SGD(self):

        batch_size = 10
        for i in range(self.loop_max):
            # random sample index
            idxs = np.random.randint(0, self.idNum, size=batch_size)
            delta = [0 for i in range(self.param_num)]
            err = 0

            for j in idxs:
                # print("*******************")
                preval = self.sigmod(np.dot(self.feats[j], self.weight))
                # print(preval)
                # print("@@@@@@@")
                for d in range(self.param_num):
                    delta[d] += (self.labels[j] - preval)*self.feats[j][d]
                    # print(delta[d])

            for j in range(self.param_num):
                self.weight[j] += self.learning_rate*delta[j]/batch_size

            for j in idxs:
                err += self.loss_function(np.dot(np.transpose(self.weight), self.feats[j]), self.labels[j])

            loss = err/batch_size
            theTime = datetime.datetime.now().strftime(self.ISOTIMEFORMAT)
            print(theTime)
            print("loss: "+ str(loss))

            if loss <= 2:
                print("****************************BEST*************************")
                print(self.weight)
                return loss

            '''
            Test the training procedure
            '''

            # print(self.feats[idxs[0]])
            # print(self.labels[idxs[0]])


        # sum_err = self.error_rate(self.idNum, self.labels, preval)
        # if i % 30 == 0:
        #     print("Iters:" + str(i) + " error:" + str(sum_err))
        #     theTime = datetime.datetime.now().strftime(self.ISOTIMEFORMAT)
        #     print(theTime)
        # err = self.labels - preval
        # delt_w = np.dot(self.feats.T, err)
        # delt_w /= self.idNum
        # self.weight += self.rate * delt_w



    def train(self):
        self.loadTrainData()
        self.idNum = len(self.feats)
        self.param_num = len(self.feats[0])
        self.initParams()
        self.ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S,f'

        weights, loss = [], []
        for i in range(self.max_iters):

            los = self.SGD()
            weights.append(self.weight)
            loss.append(los)

        index = 0
        for i in range(len(loss)):
            if loss[i] < loss[index]:
                index = i

        self.weight = weights[index]

def print_help_and_exit():
    print("usage:python3 main.py train_data.txt test_data.txt predict.txt [debug]")
    sys.exit(-1)


def parse_args():
    debug = False
    if len(sys.argv) == 2:
        if sys.argv[1] == 'debug':
            print("test mode")
            debug = True
        else:
            print_help_and_exit()
    return debug


if __name__ == "__main__":
    debug = parse_args()
    train_file =  "./data/train_data.txt"
    test_file = "./data/test_data.txt"
    # predict_file = "/projects/student/result.txt"
    predict_file = "./data/result.txt"
    lr = LR(train_file, test_file, predict_file)
    lr.train()
    lr.predict()

    if debug:
        # answer_file ="/projects/student/answer.txt"
        answer_file = "./data/answer.txt"
        f_a = open(answer_file, 'r')
        f_p = open(predict_file, 'r')
        a = []
        p = []
        lines = f_a.readlines()
        for line in lines:
            a.append(int(float(line.strip())))
        f_a.close()

        lines = f_p.readlines()
        for line in lines:
            p.append(int(float(line.strip())))
        f_p.close()

        a, p = np.array(a), np.array(p)

        print("answer mean:%.4f" % (np.mean(a)))
        print("predict mean:%.4f" % (np.mean(p)))

        tp, fp, fn ,tn = 0, 0, 0, 0
        for i in range(len(a)):
            if p[i] == a[i] and p[i] == 1:
                tp += 1

            elif p[i] == a[i]:
                tn += 1

            elif p[i] > a[i]:
                fp += 1

            elif p[i] < a[i]:
                fn += 1

        print("confusion matrix: tp: %d  fp: %d  fn: %d  tn: %d " % (tp, fp, fn, tn))



        errline = 0
        for i in range(len(a)):
            if a[i] != p[i]:
                errline += 1

        accuracy = (len(a)-errline)/len(a)
        print("accuracy:%f" %(accuracy))