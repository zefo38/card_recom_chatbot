import csv

csv_file_path = './customer_id_1/card_benefit.csv'
txt_file_path = './customer_id_1/card_benefit.txt'

card_description_template = "{card_name} 카드는 {category} 카테고리에서 {min_spend}원 이상을 썼을 경우 {discount_amount}원이 할인됩니다. 할인 한도는 {discount_limit}원이고, 할인 금액 제공 기준은 {discount_basis}원입니다. 월 이용수는 {usage_limit}입니다. 이 카드는 {card_type} 입니다.\n"

with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    with open(txt_file_path, mode='w', encoding='utf-8') as txt_file:
        for row in csv_reader:
            card_type = "신용카드" if row['Credit'] == '1' else "체크카드"
            card_description = card_description_template.format(
                card_name=row['카드명'],
                category=row['혜택 카테고리'],
                min_spend=row['이용 실적 구간'],
                discount_amount=row['할인 금액'],
                discount_limit=row['할인 한도'],
                discount_basis=row['할인 금액 제공 기준'],
                usage_limit=row['월 이용 수 '],
                card_type=card_type
            )
            txt_file.write(card_description)

txt_file_path