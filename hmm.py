# By tostq <tostq216@163.com>
# blog: blog.csdn.net/tostq
import numpy as np
from math import pi,sqrt,exp,pow,log
from numpy.linalg import det, inv
from abc import ABCMeta, abstractmethod
from sklearn import cluster

class _BaseHMM():
    """
    基本HMM虛類，需要重寫關於發射機率的相關虛函數
    n_state : 隱藏狀態的數目
    n_iter : 反覆運算次數
    x_size : 觀測值維度
    start_prob : 初始機率
    transmat_prob : 狀態轉換機率
    """
    __metaclass__ = ABCMeta  # 虛類聲明

    def __init__(self, n_state=1, x_size=1, iter=20):
        self.n_state = n_state
        self.x_size = x_size
        self.start_prob = np.ones(n_state) * (1.0 / n_state)  # 初始狀態機率
        self.transmat_prob = np.ones((n_state, n_state)) * (1.0 / n_state)  # 狀態轉換機率矩陣
        self.trained = False # 是否需要重新訓練
        self.n_iter = iter  # EM訓練的反覆運算次數

    # 初始化發射參數
    @abstractmethod
    def _init(self,X):
        pass

    # 虛函數：返回發射機率
    @abstractmethod
    def emit_prob(self, x):  # 求x在狀態k下的發射機率 P(X|Z)
        return np.array([0])

    # 虛函數
    @abstractmethod
    def generate_x(self, z): # 根據隱狀態生成觀測值x p(x|z)
        return np.array([0])

    # 虛函數：發射機率的更新
    @abstractmethod
    def emit_prob_updated(self, X, post_state):
        pass

    # 通過HMM產生序列
    def generate_seq(self, seq_length):
        X = np.zeros((seq_length, self.x_size))
        Z = np.zeros(seq_length)
        Z_pre = np.random.choice(self.n_state, 1, p=self.start_prob)  # 採樣初始狀態
        X[0] = self.generate_x(Z_pre) # 採樣得到序列第一個值
        Z[0] = Z_pre

        for i in range(seq_length):
            if i == 0: continue
            # P(Zn+1)=P(Zn+1|Zn)P(Zn)
            Z_next = np.random.choice(self.n_state, 1, p=self.transmat_prob[Z_pre,:][0])
            Z_pre = Z_next
            # P(Xn+1|Zn+1)
            X[i] = self.generate_x(Z_pre)
            Z[i] = Z_pre

        return X,Z

    # 估計序列X出現的機率
    def X_prob(self, X, Z_seq=np.array([])):
        # 狀態序列預處理
        # 判斷是否已知隱藏狀態
        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        # 向前向後傳遞因數
        _, c = self.forward(X, Z)  # P(x,z)
        # 序列的出現機率估計
        prob_X = np.sum(np.log(c))  # P(X)
        return prob_X

    # 已知當前序列預測未來（下一個）觀測值的機率
    def predict(self, X, x_next, Z_seq=np.array([]), istrain=True):
        if self.trained == False or istrain == False:  # 需要根據該序列重新訓練
            self.train(X)

        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        # 向前向後傳遞因數
        alpha, _ = self.forward(X, Z)  # P(x,z)
        prob_x_next = self.emit_prob(np.array([x_next]))*np.dot(alpha[X_length - 1],self.transmat_prob)
        return prob_x_next

    def decode(self, X, istrain=True):
        """
        利用維特比演算法，已知序列求其隱藏狀態值
        :param X: 觀測值序列
        :param istrain: 是否根據該序列進行訓練
        :return: 隱藏狀態序列
        """
        if self.trained == False or istrain == False:  # 需要根據該序列重新訓練
            self.train(X)

        X_length = len(X)  # 序列長度
        state = np.zeros(X_length)  # 隱藏狀態

        pre_state = np.zeros((X_length, self.n_state))  # 保存轉換到當前隱藏狀態的最可能的前一狀態
        max_pro_state = np.zeros((X_length, self.n_state))  # 保存傳遞到序列某位置當前狀態的最大機率

        _,c=self.forward(X,np.ones((X_length, self.n_state)))
        max_pro_state[0] = self.emit_prob(X[0]) * self.start_prob * (1/c[0]) # 初始機率

        # 前向過程
        for i in range(X_length):
            if i == 0: continue
            for k in range(self.n_state):
                prob_state = self.emit_prob(X[i])[k] * self.transmat_prob[:,k] * max_pro_state[i-1]
                max_pro_state[i][k] = np.max(prob_state)* (1/c[i])
                pre_state[i][k] = np.argmax(prob_state)

        # 後向過程
        state[X_length - 1] = np.argmax(max_pro_state[X_length - 1,:])
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            state[i] = pre_state[i + 1][int(state[i + 1])]

        return  state

    # 針對於多個序列的訓練問題
    def train_batch(self, X, Z_seq=list()):
        # 針對於多個序列的訓練問題，其實最簡單的方法是將多個序列合併成一個序列，而唯一需要調整的是初始狀態機率
        # 輸入X類型：list(array)，陣列鏈表的形式
        # 輸入Z類型: list(array)，陣列鏈表的形式，默認為空清單（即未知隱狀態情況）
        self.trained = True
        X_num = len(X) # 序列個數
        self._init(self.expand_list(X)) # 發射機率的初始化

        # 狀態序列預處理，將單個狀態轉換為1-to-k的形式
        # 判斷是否已知隱藏狀態
        if Z_seq==list():
            Z = []  # 初始化狀態序列list
            for n in range(X_num):
                Z.append(list(np.ones((len(X[n]), self.n_state))))
        else:
            Z = []
            for n in range(X_num):
                Z.append(np.zeros((len(X[n]),self.n_state)))
                for i in range(len(Z[n])):
                    Z[n][i][int(Z_seq[n][i])] = 1

        for e in range(self.n_iter):  # EM步驟反覆運算
            # 更新初始機率過程
            # E步驟
            print("iter: ", e)
            b_post_state = []  # 批量累積：狀態的後驗機率，類型list(array)
            b_post_adj_state = np.zeros((self.n_state, self.n_state)) # 批量累積：相鄰狀態的聯合後驗機率，陣列
            b_start_prob = np.zeros(self.n_state) # 批量累積初始機率
            for n in range(X_num): # 對於每個序列的處理
                X_length = len(X[n])
                alpha, c = self.forward(X[n], Z[n])  # P(x,z)
                beta = self.backward(X[n], Z[n], c)  # P(x|z)

                post_state = alpha * beta / np.sum(alpha * beta) # 歸一化！
                b_post_state.append(post_state)
                post_adj_state = np.zeros((self.n_state, self.n_state))  # 相鄰狀態的聯合後驗機率
                for i in range(X_length):
                    if i == 0: continue
                    if c[i]==0: continue
                    post_adj_state += (1 / c[i]) * np.outer(alpha[i - 1],
                                                            beta[i] * self.emit_prob(X[n][i])) * self.transmat_prob

                if np.sum(post_adj_state)!=0:
                    post_adj_state = post_adj_state/np.sum(post_adj_state)  # 歸一化！
                b_post_adj_state += post_adj_state  # 批量累積：狀態的後驗機率
                b_start_prob += b_post_state[n][0] # 批量累積初始機率

            # M步驟，估計參數，最好不要讓初始機率都為0出現，這會導致alpha也為0
            b_start_prob += 0.001*np.ones(self.n_state)
            self.start_prob = b_start_prob / np.sum(b_start_prob)
            b_post_adj_state += 0.001
            for k in range(self.n_state):
                if np.sum(b_post_adj_state[k])==0: continue
                self.transmat_prob[k] = b_post_adj_state[k] / np.sum(b_post_adj_state[k])

            self.emit_prob_updated(self.expand_list(X), self.expand_list(b_post_state))

    def expand_list(self, X):
        # 將list(array)類型的資料展開成array類型
        C = []
        for i in range(len(X)):
            C += list(X[i])
        return np.array(C)

    # 針對於單個長序列的訓練
    def train(self, X, Z_seq=np.array([])):
        # 輸入X類型：array，陣列的形式
        # 輸入Z類型: array，一維陣列的形式，預設為空清單（即未知隱狀態情況）
        self.trained = True
        X_length = len(X)
        self._init(X)

        # 狀態序列預處理
        # 判斷是否已知隱藏狀態
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))

        for e in range(self.n_iter):  # EM步驟反覆運算
            # 中間參數
            print(e, " iter")
            # E步驟
            # 向前向後傳遞因數
            alpha, c = self.forward(X, Z)  # P(x,z)
            beta = self.backward(X, Z, c)  # P(x|z)

            post_state = alpha * beta
            post_adj_state = np.zeros((self.n_state, self.n_state))  # 相鄰狀態的聯合後驗機率
            for i in range(X_length):
                if i == 0: continue
                if c[i]==0: continue
                post_adj_state += (1 / c[i])*np.outer(alpha[i - 1],beta[i]*self.emit_prob(X[i]))*self.transmat_prob

            # M步驟，估計參數
            self.start_prob = post_state[0] / np.sum(post_state[0])
            for k in range(self.n_state):
                self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])

            self.emit_prob_updated(X, post_state)

    # 求向前傳遞因數
    def forward(self, X, Z):
        X_length = len(X)
        alpha = np.zeros((X_length, self.n_state))  # P(x,z)
        alpha[0] = self.emit_prob(X[0]) * self.start_prob * Z[0] # 初始值
        # 歸一化因數
        c = np.zeros(X_length)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]
        # 遞迴傳遞
        for i in range(X_length):
            if i == 0: continue
            alpha[i] = self.emit_prob(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
            c[i] = np.sum(alpha[i])
            if c[i]==0: continue
            alpha[i] = alpha[i] / c[i]

        return alpha, c

    # 求向後傳遞因數
    def backward(self, X, Z, c):
        X_length = len(X)
        beta = np.zeros((X_length, self.n_state))  # P(x|z)
        beta[X_length - 1] = np.ones((self.n_state))
        # 遞迴傳遞
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.transmat_prob.T) * Z[i]
            if c[i+1]==0: continue
            beta[i] = beta[i] / c[i + 1]

        return beta

# 二元高斯分佈函數
def gauss2D(x, mean, cov):
    # x, mean, cov均為numpy.array類型
    z = -np.dot(np.dot((x-mean).T,inv(cov)),(x-mean))/2.0
    temp = pow(sqrt(2.0*pi),len(x))*sqrt(det(cov))
    return (1.0/temp)*exp(z)

class GaussianHMM(_BaseHMM):
    """
    發射機率為高斯分佈的HMM
    參數：
    emit_means: 高斯發射機率的均值
    emit_covars: 高斯發射機率的方差
    """
    def __init__(self, n_state=1, x_size=1, iter=20):
        _BaseHMM.__init__(self, n_state=n_state, x_size=x_size, iter=iter)
        self.emit_means = np.zeros((n_state, x_size))      # 高斯分佈的發射機率均值
        self.emit_covars = np.zeros((n_state, x_size, x_size)) # 高斯分佈的發射機率協方差
        for i in range(n_state): self.emit_covars[i] = np.eye(x_size)  # 初始化為均值為0，方差為1的高斯分佈函數

    def _init(self,X):
        # 通過K均值聚類，確定狀態初始值
        mean_kmeans = cluster.KMeans(n_clusters=self.n_state)
        mean_kmeans.fit(X)
        self.emit_means = mean_kmeans.cluster_centers_
        for i in range(self.n_state):
            self.emit_covars[i] = np.cov(X.T) + 0.01 * np.eye(len(X[0]))

    def emit_prob(self, x): # 求x在狀態k下的發射機率
        prob = np.zeros((self.n_state))
        for i in range(self.n_state):
            prob[i]=gauss2D(x,self.emit_means[i],self.emit_covars[i])
        return prob

    def generate_x(self, z): # 根據狀態生成x p(x|z)
        return np.random.multivariate_normal(self.emit_means[z][0],self.emit_covars[z][0],1)

    def emit_prob_updated(self, X, post_state): # 更新發射機率
        for k in range(self.n_state):
            for j in range(self.x_size):
                self.emit_means[k][j] = np.sum(post_state[:,k] *X[:,j]) / np.sum(post_state[:,k])

            X_cov = np.dot((X-self.emit_means[k]).T, (post_state[:,k]*(X-self.emit_means[k]).T).T)
            self.emit_covars[k] = X_cov / np.sum(post_state[:,k])
            if det(self.emit_covars[k]) == 0: # 對奇異矩陣的處理
                self.emit_covars[k] = self.emit_covars[k] + 0.01*np.eye(len(X[0]))


class DiscreteHMM(_BaseHMM):
    """
    發射機率為離散分佈的HMM
    參數：
    emit_prob : 離散機率分佈
    x_num：表示觀測值的種類
    此時觀測值大小x_size默認為1
    """
    def __init__(self, n_state=1, x_num=1, iter=20):
        _BaseHMM.__init__(self, n_state=n_state, x_size=1, iter=iter)
        self.emission_prob = np.ones((n_state, x_num)) * (1.0/x_num)  # 初始化發射機率均值
        self.x_num = x_num

    def _init(self, X):
        self.emission_prob = np.random.random(size=(self.n_state,self.x_num))
        for k in range(self.n_state):
            self.emission_prob[k] = self.emission_prob[k]/np.sum(self.emission_prob[k])

    def emit_prob(self, x): # 求x在狀態k下的發射機率
        prob = np.zeros(self.n_state)
        for i in range(self.n_state): prob[i]=self.emission_prob[i][int(x[0])]
        return prob

    def generate_x(self, z): # 根據狀態生成x p(x|z)
        return np.random.choice(self.x_num, 1, p=self.emission_prob[z][0])

    def emit_prob_updated(self, X, post_state): # 更新發射機率
        self.emission_prob = np.zeros((self.n_state, self.x_num))
        X_length = len(X)
        for n in range(X_length):
            self.emission_prob[:,int(X[n])] += post_state[n]

        self.emission_prob+= 0.1/self.x_num
        for k in range(self.n_state):
            if np.sum(post_state[:,k])==0: continue
            self.emission_prob[k] = self.emission_prob[k]/np.sum(post_state[:,k])
