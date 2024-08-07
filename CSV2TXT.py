class csv2txt():
    def __init__(self, data, save_path):
        self.data = data
        self.save_path = save_path
        
    def transform(self):
        df = self.data
        transactions = []
        for index, row in df.iterrows():
            date = row['날짜']
            customer_id = row['고객번호']
            for category in df.columns[2:]:
                amount = row[category]
                if amount >= 0:
                    transaction_txt = f"고객번호 {customer_id}번 고객님은 {date}에 {category}에서 {amount}원을 썼습니다"
                    transactions.append(transaction_txt)
                else :
                    transaction_txt = "예기치 못한 오류로 해당 데이터를 불러오지 못했습니다"
                    transactions.append(transaction_txt)
        
        with open(self.save_path, 'w', encoding = 'utf-8') as file:
            for transaction in transactions:
                file.write(transaction + '\n')
        return file
