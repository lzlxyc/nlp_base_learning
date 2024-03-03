import pandas as pd


def get_data():
    # 创建包含数据的字典
    df = pd.read_csv('F:/learning/nlp_base_learning/data/A榜-训练集_海上风电预测_气象变量及实际功率数据.csv',encoding='gbk')
    df = df[df['站点编号']=='f1']
    df['时间'] = pd.to_datetime(df['时间'])

    df['mooth'] = df['时间'].dt.month
    df['hour'] = df['时间'].dt.hour

    print(df.columns)
    # 打印输出整理后的数据框
    print(df.head(5))
    return df

if __name__ == '__main__':
    get_data()