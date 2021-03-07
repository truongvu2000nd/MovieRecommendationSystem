import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os

# #####################
# Input
path = os.path.join(os.getcwd() + '/ML Datamining/ml-100K/')
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_train = pd.read_csv(path + 'ua.base', sep='\t', names=r_cols, encoding='latin-1').values
ratings_test = pd.read_csv(path + 'ua.test', sep='\t', names=r_cols, encoding='latin-1').values

# #####################
# Model
class CollaborativeFiltering:
    def __init__(self, num_neighbors, dist_func=cosine_similarity, uuCF=1):
        """ hàm khởi tạo (constructor) của class
            num_neighbors: số lượng hàng xóm để xét
            dist_func: hàm tính khoảng cách. Default là khoảng cách coisn
            uuCF: sử dụng user-user hay item-item (Default là 1 tức là user-user)
        """
        super().__init__()
        self.num_neighbors = num_neighbors
        self.dist_func = dist_func
        self.uuCF = uuCF

    def _normalize_data(self, data):
        """Normalize dữ liệu _data

        Args:
        Return: Ma trận đã được chuẩn hóa
        """

        user = data[:, 0].astype(np.int32)
        item = data[:, 1].astype(np.int32)
        rating = data[:, 2].astype(np.float64)

        num_users = np.max(user) + 1
        num_items = np.max(item) + 1

        self.mu = np.zeros(num_users)
        # tìm các rating của user
        for n in range(num_users):
            id_user = np.where(user == n)[0]            # tìm kiếm user == n
            rating_mean = np.mean(rating[id_user])       # tính giá trị trung bình(mean) của rating

            if np.isnan(rating_mean):
                rating_mean = 0
            self.mu[n] = rating_mean

            rating[id_user] = rating[id_user] - self.mu[n]

        self.utility_data = csr_matrix( (rating, (user, item)) )

    def fit(self, data):
        self._normalize_data(data)
        self.Matrix_Cosine = self.dist_func(self.utility_data, self.utility_data)
        print(self.Matrix_Cosine.shape)
        
    def predict(self, user, item):
        pass

# ###################
# Evaluation and Test
cf = CollaborativeFiltering(num_neighbors=10)
cf.fit(ratings_train)
