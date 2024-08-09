from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import pacmap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    
      def get_sum_score(self, cluster_mean, card_data, weights):
            return sum(cluster_mean[feature] * card_data[feature] * weights.get(feature, 1) for feature in card_data.index[2:])
    
      def rec_n(self, cluster_data, card_data, weights, n = 3):
            cluster_mean = cluster_data.groupby(['cluster']).mean()
            recs = {}
            for cluster, mean_values in cluster_mean.iterrows():
                  scores = []
                  for _, card in card_data.iterrows():
                      score = self.get_sum_score(mean_values, card, weights)
                      scores.append((card['CardID'], card['CardName'], score))
                  top_n_card = sorted(scores, key = lambda x : x[2], reverse = True)[:n]
                  recs[cluster] = top_n_card
            return recs

      def get_cosine_sim(self, cluster_mean, indi_mean):
            cluster_mean_array = cluster_mean.values
            indi_mean_array = indi_mean.values.reshape(1, -1)
            cos_sim = cosine_similarity(cluster_mean_array, indi_mean_array).flatten()
            most_sim = np.argmax(cos_sim)
            return most_sim
