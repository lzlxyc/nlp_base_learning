import pandas as pd
from tqdm import tqdm
from inference_pointwise import test_inference

data_path = './data/comment_classify2/test.csv'


if __name__ == "__main__":
    df = pd.read_csv(data_path,encoding='utf-8')
    res_list = []

    len_df = len(df)
    for idx in tqdm(range(len_df)):
        description = df.iloc[idx]['description']
        text = df.iloc[idx]['text']
        res = test_inference(
            description,
            text,
            max_seq_len=128
        )
        # print(res,description,text[:10])

        res_list.append(res)
    df['label'] = res_list
    df.to_csv('./data/comment_classify2/test_res2.csv',index=0,encoding='utf-8-sig')
    df[['task_id','record_id','label']].to_csv('./data/comment_classify2/sample_res2.csv',index=0,encoding='utf-8-sig')
