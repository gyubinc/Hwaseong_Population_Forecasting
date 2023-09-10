def preprocessing(df, window_size, step):
    # 날짜를 정수로 변경
    #df['월별'] = pd.to_datetime(df['월별'])
    #df['월별'] = df['월별'].values.astype(float)

    
    # 문자열 데이터 삭제
    for column in df.columns:
        if df[column].dtype == 'object':
            df = df.drop(column, axis=1)
    df = df.astype(float)
    df = df.dropna()
    # train_df = df.iloc[:-12, :]
    train_df = df.iloc[:-step, :]
    valid_df = df.iloc[-window_size-step:, :]
    return train_df, valid_df

def inference_preprocessing(df, window_size, step):
    # 날짜를 정수로 변경
    #df['월별'] = pd.to_datetime(df['월별'])
    #df['월별'] = df['월별'].values.astype(float)

    
    # 문자열 데이터 삭제
    for column in df.columns:
        if df[column].dtype == 'object':
            df = df.drop(column, axis=1)
    df = df.astype(float)
    df = df.dropna()
    # train_df = df.iloc[:-12, :]
    train_df = df
    valid_df = df.iloc[-window_size:, :]
    return train_df, valid_df