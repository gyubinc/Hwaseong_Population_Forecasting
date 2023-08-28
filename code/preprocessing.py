import pandas as pd
def preprocessing(df, window_size):
    # 날짜를 정수로 변경
    #df['월별'] = pd.to_datetime(df['월별'])
    df['월별'] = df['월별'].values.astype(float)

    
    # 문자열 데이터 삭제
    for column in df.columns:
        if df[column].dtype == 'object':
            df = df.drop(column, axis=1)
    df = df.astype(float)
    df = df.dropna()
    train_df = df.iloc[:-12, :]
    valid_df = df.iloc[-12-window_size:, :]
    return train_df, valid_df