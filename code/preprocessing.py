def preprocessing(df, window_size, step):
    # 문자열 데이터 삭제
    for column in df.columns:
        if df[column].dtype == 'object':
            df = df.drop(column, axis=1)
    
    # 숫자형 데이터 float 변형
    df = df.astype(float)
    
    # 빈 데이터 삭제
    df = df.dropna()
    
    # train / valid 분리
    train_df = df.iloc[:-step, :]
    valid_df = df.iloc[-window_size-step:, :]
    return train_df, valid_df

def inference_preprocessing(df, window_size):
    # 문자열 데이터 삭제
    for column in df.columns:
        if df[column].dtype == 'object':
            df = df.drop(column, axis=1)
            
    # 숫자형 데이터 float 변형
    df = df.astype(float)
    
    # 빈 데이터 삭제
    df = df.dropna()
    
    # train / valid 분리 (inference)
    train_df = df
    valid_df = df.iloc[-window_size:, :]
    return train_df, valid_df