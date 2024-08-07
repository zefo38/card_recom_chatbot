from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import pacmap

class recsys():
    def __init__(self, data, random_state):
            self.data = data
            self.random_state = random_state

    def preprocessing(self, n_components, n_neighbors, MN_ratio, FP_ratio):
          scaler = StandardScaler()
          scaled_data = scaler(self.data)
          embedding = pacmap.PaCMAP(n_components = n_components, n_neighbors = n_neighbors, MN_ratio = MN_ratio, FP_ratio = FP_ratio)
          pac_data = embedding.fit_transform(scaled_data)
          return pac_data
    
    def clustering(self, n_cluster, pre_data):
          c_data = self.data.copy()
          km = KMeans(n_clusters = n_cluster)
          clusters = km.fit_transform(pre_data)
          c_data['cluster'] = km.fit_transform(pre_data)
          return clusters, c_data
    
    def get_sum_score(self, cluster_data, )