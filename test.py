import boto3
import pandas as pd

file_path = "data/train.csv"
data = pd.read_csv(file_path)
length = data.shape[0]
translate = boto3.client(service_name='translate', region_name='us-east-2', use_ssl=True)
re_trans = boto3.client(service_name='translate', region_name='us-east-2', use_ssl=True)
for idx, row in data.iterrows():
    result1 = translate.translate_text(Text=row["text_a"], SourceLanguageCode="zh", TargetLanguageCode="en")
    result2 = translate.translate_text(
        Text=result1.get('TranslatedText'),
        SourceLanguageCode="en",
        TargetLanguageCode="zh")
    text_a = result2.get('TranslatedText')
    result1 = translate.translate_text(
        Text=row["text_b"],
        SourceLanguageCode="zh",
        TargetLanguageCode="en")
    result2 = translate.translate_text(
        Text=result1.get('TranslatedText'),
        SourceLanguageCode="en",
        TargetLanguageCode="zh")
    text_b = result2.get('TranslatedText')
    data.loc[length + idx] = [text_a, text_b, row['part'], row['label']]

print(data.shape)
file_path = "data/aug_train.csv"
data.to_csv(file_path, index=False, encoding="utf8")