import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_data():
    df_transactions = pd.read_csv('./card/card_transaction2.csv')
    credit = pd.read_csv('./card/card2.csv')
    return df_transactions, credit


def card_recommend():
    df_transactions, credit = load_data()

    # 데이터 전처리
    df_sum = df_transactions.drop('날짜', axis=1)
    df_sum = df_sum.groupby(['고객번호']).sum()
    df_sum2 = df_sum.drop(['부동산', '여행', '자동차', '항공사', '기기', '기타'], axis=1)

    @st.cache_resource
    def umap():
        import umap
        from sklearn.preprocessing import MinMaxScaler
        # 데이터 스케일링 - 정규화
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_sum2)

        # UMAP 모델 생성
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
        umap_df = umap_model.fit_transform(scaled_data)
        return umap_df

    umap_df = umap()
        # K-means 클러스터링 적용
    kmeans = KMeans(n_clusters=3)
    df_sum2['cluster'] = kmeans.fit_predict(umap_df)
    cluster_means = df_sum2.groupby('cluster').mean()

    # 가중치 설정
    weights = {
        '교통': 1.0,
        '온라인결제': 1.3,
        '독서실': 1.0,
        '병원': 1.1,
        '쇼핑': 1.3,
        '문화': 1.0,
        '음식점': 0.9,
        '서점': 1.2,
        '카페': 0.9,
        '여행': 1.0,
        '동물병원': 1.0,
        '영화': 1.2,
        '스포츠': 1.0,
        '미용실': 1.0,
        '편의점': 0.9,
        '학원': 1.0,
        '이동통신': 1.0,
        '세탁소': 1.2,
        '숙박': 1.0,
        '배달': 1.0,
        '해외결제': 1.0,
        '대형마트': 1.0,
        'OTT구독': 1.0,
        '해외직구': 1.0,
        '테마파크': 1.2
    }

    card_df1 = pd.DataFrame({
        'CardID': [10, 11, 12, 13, 14, 15, 16, 17],
        'CardName': ["MyWESH_먹는데진심", "MyWESH_노는데진심", "MyWESH_관리에진심", "다담_생활", "다담_교육", "다담_쇼핑", "다담_직장인", "다담_레저"],
        '쇼핑': [0, 1, 0, 0, 0, 1, 0, 0],
        '카페': [0, 1, 0, 0, 0, 0, 1, 0],
        '음식점': [1, 0, 0, 0, 0, 0, 0, 0],
        '숙박': [0, 0, 0, 0, 0, 0, 0, 0],
        '서점': [0, 0, 1, 0, 1, 0, 0, 0],
        '온라인결제': [0, 0, 0, 0, 0, 1, 0, 0],
        '교통': [0, 1, 0, 0, 0, 0, 0, 0],
        '미용실': [0, 0, 1, 1, 0, 0, 0, 0],
        '병원': [0, 0, 0, 0, 0, 0, 0, 0],
        '주유소': [0, 0, 0, 1, 0, 0, 0, 1],
        '화장품': [0, 0, 1, 0, 0, 0, 0, 0],
        '편의점': [0, 1, 1, 1, 0, 1, 1, 0],
        '주차장': [1, 0, 0, 0, 0, 0, 0, 0],
        '문화': [0, 1, 1, 1, 0, 0, 0, 1],
        '이동통신': [1, 0, 1, 1, 1, 1, 0, 0],
        '학원': [0, 0, 0, 0, 1, 0, 0, 0],
        '스포츠': [0, 0, 1, 0, 0, 0, 0, 1],
        '동물병원': [0, 0, 0, 0, 0, 0, 0, 0],
        '세탁소': [0, 0, 0, 0, 1, 0, 0, 0],
        '영화': [0, 1, 0, 1, 0, 0, 0, 0],
        '해외직구': [0, 0, 0, 0, 0, 0, 0, 1],
        '택시': [0, 1, 0, 0, 0, 0, 0, 0],
        '테마파크': [0, 0, 0, 1, 0, 0, 0, 0],
        '배달': [1, 0, 0, 0, 0, 0, 0, 0],
        '해외결제': [0, 0, 0, 1, 1, 1, 1, 1],
        '대형마트': [0, 0, 0, 1, 0, 0, 0, 0],
        'OTT구독': [1, 1, 1, 0, 0, 0, 0, 0]
    })

    card_df2 = pd.DataFrame({
        'CardID': [18, 19, 20, 21],
        'CardName': ["EasyAll티타늄_A", "EasyAll티타늄_B", "EasyAll티타늄_C", "EasyAll티타늄_D"],
        '쇼핑': [0, 0, 1, 1],
        '카페': [0, 0, 0, 1],
        '음식점': [1, 0, 0, 0],
        '숙박': [0, 0, 0, 0],
        '서점': [0, 0, 0, 0],
        '온라인결제': [0, 1, 0, 0],
        '교통': [0, 1, 0, 0],
        '미용실': [0, 0, 0, 1],
        '병원': [1, 0, 0, 0],
        '주유소': [0, 1, 0, 1],
        '화장품': [0, 0, 1, 0],
        '편의점': [0, 1, 0, 0],
        '주차장': [0, 0, 0, 0],
        '문화': [0, 0, 0, 1],
        '이동통신': [0, 1, 0, 1],
        '학원': [1, 0, 0, 0],
        '스포츠': [0, 0, 1, 0],
        '동물병원': [0, 0, 0, 0],
        '세탁소': [0, 0, 0, 0],
        '영화': [0, 0, 0, 1],
        # '부동산' : [0,0,0,0],
        # '자동차' : [0,0,0,0],
        '해외직구': [0, 0, 0, 0],
        '택시': [0, 0, 0, 1],
        '테마파크': [0, 0, 0, 0],
        # '항공사' : [0,0,0,0],
        '배달': [0, 0, 0, 1],
        '해외결제': [1, 0, 0, 0],
        '대형마트': [0, 1, 0, 0],
        'OTT구독': [0, 0, 0, 1]

    })

    card_df3 = pd.DataFrame({
        'CardID': [22, 23],
        'CardName': ['굿데이올림카드', 'BeVV카드'],
        '쇼핑': [0, 1],
        '카페': [0, 0],
        '음식점': [0, 0],
        '숙박': [0, 1],
        '서점': [0, 0],
        '온라인결제': [0, 0],
        '교통': [1, 0],
        '미용실': [0, 0],
        '병원': [1, 0],
        '주유소': [1, 0],
        '화장품': [0, 0],
        '편의점': [1, 0],
        '주차장': [0, 0],
        '문화': [0, 0],
        '이동통신': [1, 0],
        '학원': [1, 0],
        '스포츠': [0, 1],
        '동물병원': [0, 0],
        '세탁소': [0, 0],
        '영화': [0, 1],
        # '부동산' : [0,0],
        # '자동차' : [0,0],
        '해외직구': [1, 0],
        '택시': [0, 0],
        '테마파크': [0, 0],
        # '항공사' : [0,0],
        '배달': [0, 0],
        '해외결제': [1, 0],
        '대형마트': [1, 0],
        'OTT구독': [0, 0]

    })

    card_df = pd.concat([card_df1, card_df2, card_df3], axis=0)

    def weighted_sum_score(cluster_mean, card, weights):
        return sum(cluster_mean[feature] * card[feature] * weights.get(feature, 1) for feature in card.index[2:])

    def recommend_top_n_cards(cluster_mean, card_df, weights, n=3):
        recommendations = {}
        for cluster, mean_values in cluster_mean.iterrows():
            scores = []
            for _, card in card_df.iterrows():
                score = weighted_sum_score(mean_values, card, weights)
                scores.append((card['CardID'], card['CardName'], score))
            top_n_cards = sorted(scores, key=lambda x: x[2], reverse=True)[:n]
            recommendations[cluster] = top_n_cards
        return recommendations

    top_n_recommendations = recommend_top_n_cards(cluster_means, card_df, weights, n=3)

    # 특정 사용자에게 카드 추천
    individual_card_df = df_sum2.loc[10]
    individual_df = individual_card_df.to_frame().T
    individual_mean = individual_df.drop('cluster', axis=1)

    cluster_means_array = cluster_means.values
    individual_mean_array = individual_mean.values.reshape(1, -1)

    similarity_scores = cosine_similarity(cluster_means_array, individual_mean_array).flatten()
    most_similar_cluster = np.argmax(similarity_scores)

    card3 = []
    for i in range(0, 3):
        card3.append(top_n_recommendations[most_similar_cluster][i][1])
    return card3


