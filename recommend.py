import yaml
import streamlit as st
import pandas as pd

def data():
    df_transactions = pd.read_csv('./card/card_transaction2.csv')
    card_benefit_df = pd.read_csv('./card/card2.csv')
    df_transactions = df_transactions.loc[df_transactions['고객번호'] == 40]

    df_transactions['날짜'] = pd.to_datetime(df_transactions['날짜'])

    card_benefit = card_benefit_df.groupby(['카드명', '혜택 카테고리']).mean().reset_index()

    card_benefit.drop(card_benefit.loc[card_benefit['혜택 카테고리'].isin(['음식점', '편의점'])].index, inplace=True)

    # 카테고리별 이용 횟수를 계산하기 위한 데이터프레임 생성
    df_counts = pd.DataFrame(columns=['고객번호', '카드명', '혜택 카테고리', '이용 횟수', 'CREDIT분류'])

    # 각 고객에 대해 혜택 카테고리별 이용 횟수 계산
    rows_to_add = []
    for idx, row in df_transactions.iterrows():
        고객번호 = row['고객번호']
        for benefit_idx, benefit_row in card_benefit.iterrows():
            카드명 = benefit_row['카드명']
            혜택_카테고리 = benefit_row['혜택 카테고리']
            신용카드분류 = benefit_row['Credit']
            if row.get(혜택_카테고리, 0) > 0:
                rows_to_add.append({
                    '고객번호': 고객번호,
                    '카드명': 카드명,
                    'CREDIT분류': 신용카드분류,
                    '혜택 카테고리': 혜택_카테고리,
                    '이용 횟수': 1
                })

    # DataFrame으로 변환 후 concat

    df_counts = pd.concat([df_counts, pd.DataFrame(rows_to_add)], ignore_index=True)

    # 고객별 카드 혜택별 이용 횟수
    grouped_df = df_counts.groupby(['고객번호', '카드명', '혜택 카테고리', 'CREDIT분류']).sum().reset_index()
    grouped_df.columns = ['고객번호', '카드명', '혜택 카테고리', 'CREDIT분류', '이용 횟수']

    grouped_df['이용 횟수'] = pd.to_numeric(grouped_df['이용 횟수'])

    # 피벗 테이블 생성
    pivot_df = grouped_df.pivot_table(index=['고객번호', '카드명', 'CREDIT분류'], columns='혜택 카테고리', values='이용 횟수',
                                      fill_value=0)

    # 컬럼 이름 정렬 (카테고리 순서 유지)
    pivot_df = pivot_df.sort_index(axis=1)

    row_sums = pivot_df.sum(axis=1)

    # 총합을 데이터프레임으로 변환
    row_sums_df = row_sums.reset_index(name='총합')

    # 고객번호와 총합으로 정렬
    sorted_row_sums = row_sums_df.sort_values(by=['고객번호', '총합'], ascending=[False, False])

    sum_df = sorted_row_sums.sort_values(['고객번호', '총합'], ascending=[True, False])

    # 카드명 리스트를 생성
    card_list = ['다담_생활', '다담_직장인', '다담_쇼핑', '다담_레저', '다담_교육']

    for card in card_list:
        array_1 = (
                sum_df.loc[sum_df['카드명'] == '다담']
                .groupby(['고객번호', '카드명', 'CREDIT분류'])
                .sum()['총합'].values
                + sum_df.loc[sum_df['카드명'] == card]
                .groupby(['고객번호', '카드명', 'CREDIT분류'])
                .sum()['총합'].values
        )

        dd1 = array_1.tolist()
        sum_df.loc[sum_df['카드명'] == card, '총합'] = dd1
    sum_df = sum_df.loc[sum_df['카드명'] != '다담']

    cards1 = sum_df.sort_values('총합', ascending=False).head(3)['카드명'].tolist()

    benefit = grouped_df[grouped_df['카드명'].isin(cards1)].sort_values('이용 횟수', ascending=False)

    bene = []
    for card in cards1:
        bene.append(benefit.loc[benefit['카드명'] == card].head(5)['혜택 카테고리'].reset_index(drop=True))

    for i in range(len(cards1)):
        if '다담' in cards1[i]:
            cards1[i] = '다담'

    return cards1, bene