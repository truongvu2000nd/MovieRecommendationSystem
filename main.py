import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


# #####################
# Input


# #####################
# Model
class CollaborativeFiltering():
    """Class CF
    """
    def __init__(self, num_neighbors, dist_func=cosine_similarity):
        """constructor cho class

        Args:
            num_neighbors (int): số lượng hàng xóm trong bước dự đoán
            dist_func (function, optional): hàm tính độ tương quan giữa 2 người hoặc 2 phim. Defaults to cosine_similarity.
        """
        super().__init__()
        self.num_neighbors = num_neighbors
        self.dist_func = dist_func

    def _normalize_data(self, utility_data):
        """Normalize dữ liệu utility_data
        Điền vào các rating còn thiếu
        Trừ các rating đi giá trị trung bình

        Args:
            utility_data (sparse matrix): 3 trận thưa 3 chiều (user_id, movie_id, rating)

        Return: Ma trận đã được chuẩn hóa
        """

    def fit(self, utility_data):
        """Fit model theo data

        Args:
            utility_data (sparse matrix): 3 trận thưa 3 chiều (user_id, movie_id, rating)
        """


    def predict(self, user, item):
        """Dự đoán rating của user cho item
        """


# ###################
# Evaluation and Test