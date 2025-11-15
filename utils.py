# ========================================================================
# utils.py
# 
# 說明：
# 本檔案整合了所有資料讀取 (Data Loading) 和預處理 (Preprocessing) 的共用函式。
#
# 主要功能分為：
# 1. 資料讀取 (Get Data): 從 CSV 檔案讀取不同類型的資料。
# 2. 資料豐富化 (Enrich Data): 替既有的 DataFrame 加入新欄位 (如日期、性別)。
# 3. 資料清理 (Clean Data): 移除空值、處理極端值 (Outliers)。
# ========================================================================

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datetime import date

# --- 全域路徑設定 (Global Path Setup) ---
# 建議將路徑設定放在這裡，方便統一管理
FIG_PATH = Path('/home/cmc1503/Desktop/JD_exploration/figures')
DATA_PATH = Path('/home/cmc1503/Desktop/JD_exploration/data')

# 設定 Pandas 顯示格式
pd.options.display.float_format = '{:.4f}'.format


# ===================================
# 1. 資料讀取 (Get Data)
# ===================================

def get_track_event(start=1, end=1) -> pd.DataFrame:
    """
    讀取「週次」的 'track_event' 資料。
    資料被分片 (sharded) 儲存 (例如 week01_%0.csv ~ week01_%9.csv)。
    
    Args:
        start (int): 開始週次 (包含).
        end (int): 結束週次 (包含).
    
    Returns:
        pd.DataFrame: 包含指定區間所有 'track_event' 資料的 DataFrame.
    """
    data = pd.DataFrame()
    track_event_path = DATA_PATH / 'track_event'
    
    for week in range(start, end + 1):
        df_week = pd.DataFrame()
        week_str = f'week{week:02d}' # 格式化為 week01, week02...
        
        for i in range(10):
            try:
                file_name = f"{week_str}_%{i}.csv"
                temp = pd.read_csv(track_event_path / file_name)
                df_week = pd.concat([df_week, temp])
            except FileNotFoundError:
                print(f"Warning: 檔案 {file_name} 不存在，已跳過。")
                continue
        
        if not df_week.empty:
            df_week["week"] = week
            data = pd.concat([data, df_week])
            
    return data


def get_events(group: str) -> pd.DataFrame:
    """
    讀取「月次」的 'events' 資料 (例如 month3_A.csv)。
    
    Args:
        group (str): 實驗組別 ('A', 'B', 'C', 'D').
    
    Returns:
        pd.DataFrame: 包含指定月份 'events' 資料的 DataFrame.
    """
    data = pd.DataFrame()
    events_path = DATA_PATH / 'events'
    
    for month in range(3, 8): # 月份是 3 - 7
        try:
            file_name = f"month{month}_{group}.csv"
            df = pd.read_csv(events_path / file_name)
            data = pd.concat([data, df])
        except FileNotFoundError:
            print(f"Warning: 檔案 {file_name} 不存在，已跳過。")
            continue
            
    return data


def get_background(columns: list = None, gender: str = 'F', last_login=202307) -> pd.DataFrame:
    """
    讀取使用者背景資料 (users_females.csv 或 users_males.csv)。
    
    Args:
        columns (list, optional): 需要讀取的欄位. 預設為 None (讀取全部).
        gender (str, optional): 'F' (女性) 或 'M' (男性). 預設為 'F'.
        last_logint (int): YYYYMM 格式，只保留 'last_login_ymd' >= 此日期的帳號。

    Returns:
        pd.DataFrame: 使用者背景資料.
    """
    if gender == 'F':
        file_name = 'users_females.csv'
    else:
        file_name = 'users_males.csv'
        
    data = pd.read_csv(DATA_PATH / file_name)

    # 只保留 last_login_ymd 在202307後的帳號
    filt = data['last_login_ymd'] >= last_login

    if columns:
        return data[filt][columns]
    else:
        return data[filt]

# ===================================
# 2. 資料豐富化 (Enrich Data)
# ===================================

def insert_date(data: pd.DataFrame, ts_col: str = 'ts') -> pd.DataFrame:
    """
    將 timestamp 欄位轉換為 'datetime' 和 'date' 欄位。
    
    Args:
        data (pd.DataFrame): 原始 DataFrame.
        ts_col (str, optional): timestamp 欄位的名稱. 預設為 'ts'.
    """
    data['datetime'] = data[ts_col].apply(lambda x: datetime.fromtimestamp(x))
    data['date'] = data['datetime'].dt.date
    return data


def insert_gender(data: pd.DataFrame, background_df: pd.DataFrame) -> pd.DataFrame:
    """
    根據背景資料，將 'gender' 欄位併入主 DataFrame。
    
    Args:
        data (pd.DataFrame): 主要資料 (例如 track_event).
        background_df: users_males.csv or users_females.csv 或 combine 過後的
        gender: 要加入的性別
    """
    # 建立 uCode 到 gender 的對應字典
    gender_dict = dict(zip(background_df.uCode, background_df.gender))
    data['gender'] = data['uCode'].map(gender_dict)
    return data

def insert_group(data):
    """
    併入實驗組別 ('group') 資訊。
    ** 效能注意：此函式*內部*讀取了 'fem_treat_arms_sep_jan_20w.csv' **

    Args:
        data (pd.DataFrame): 要修改的 DataFrame (需有 'uCode')。
    """
    df = pd.read_csv(DATA_PATH / 'fem_treat_arms_sep_jan_20w.csv')
    group_dict = dict(zip(df['uCode'], df['divide']))
    data['group'] = data['uCode'].map(group_dict)
    return data

def insert_background(data, background_df, col, gender):
    """
    從 background data 併入*任何*指定的欄位 (col)。

    Args:
        data (pd.DataFrame): 要修改的 DataFrame。
        background_df: users_males.csv or users_females.csv 或 combine 過後的
        col (str): 要併入的欄位名稱 (例如 'age', 'city')。
        gender (str): 'F' 或 'M'。
        last_login (int): 傳給 get_background 的篩選參數。
    """
    background_dict = dict(zip(background_df['uCode'], background_df[col]))
    
    if gender == 'F':
        # 如果是 'F'，併入 'uCode' (主動方)
        data['uCode_{}'.format(col)] = data['uCode'].map(background_dict)
    else:
        # 如果是 'M'，併入 'target_uCode' (被動方)
        data['target_{}'.format(col)] = data['target_uCode'].map(background_dict)
    return

# ===================================
# 3. 資料清理 (Clean Data)
# ===================================

def get_prob_accounts_by_BD(background_df) -> pd.Series:
    """
    (來自 all_backgrounds.ipynb)
    找出 '有問題的帳號' (Problematic Accounts) - 依據生日。
    條件：生日 < 1954-01-01 或 >= 2007-01-01 或為空值 (age <= 0 or > 70)

    Args:
        background_df: users_males.csv or users_females.csv 或 combine 過後的

    Returns:
        pd.Series: 有問題帳號的 'uCode' 列表。
    """
    # ** 效能注意：此函式*內部*呼叫了 get_background **

    background_df['birthday'] = pd.to_datetime(background_df.birthday, errors="coerce", format="%Y-%m-%d")
    low_filt = (background_df.birthday < np.datetime64(date(1954, 1, 1)))
    up_filt = (background_df.birthday >= np.datetime64(date(2007, 1, 1)))
    na_filt = background_df.birthday.isna()
    prob_accounts = background_df[low_filt | up_filt | na_filt]['uCode']
    return prob_accounts

def get_prob_accounts_by_Yes(data, threshold=0.999) -> pd.Series:
    """
    找出 '有問題的帳號' - 依據送出 'interestYes' 的數量。
    條件：女性，且 'interestYes' 數量 > 全體女性的 99.9 百分位數。

    Args:
        data (pd.DataFrame): *主要*的 track_event 資料 (需包含 'gender', 'act', 'uCode')。
        threshold (float): 百分位數閾值 (預設 0.999)。

    Returns:
        pd.Series: 有問題帳號的 'uCode' 列表。
    """
    gender_filt = (data['gender'] == 'F')
    act_filt = (data['act'] == 'interestYes')
    cnt_Yes = data[gender_filt & act_filt][['uCode', 'act']].groupby('uCode').count()
    
    prob_accounts = cnt_Yes[cnt_Yes['act'] > cnt_Yes['act'].quantile(threshold)].reset_index()['uCode']
    return prob_accounts

def drop_prob_accounts(data, background_df, threshold=0.999):
    """
    從主資料框 (df) 中移除所有 "有問題的帳號"。
    這是一個工作流程 (workflow) 函式，整合了多個清理步驟。
    
    移除條件 (任一成立)：
    1. uCode 或 target_uCode 的生日有問題 (來自 get_prob_accounts_by_BD)。
    (生日在 1954/1/1 之前 或 2007/1/1 之後)
    2. uCode (女性) 發送 'interestYes' 過多 (來自 get_prob_accounts_by_Yes)。
    (總按讚數在該性別當中的 99.9% 以上)

    Args:
        data (pd.DataFrame): *主要*的 track_event 資料。
        background_df: users_males.csv or users_females.csv 或 combine 過後的
        threshold (float): 傳給 get_prob_accounts_by_Yes 的閾值。

    Returns:
        pd.DataFrame: 清理過後的 DataFrame。
    """    
    t1 = time.process_time()
    # ** 效能注意：此處呼叫了 get_background 兩次 (F 和 M) **
    prob_accounts_by_BD = get_prob_accounts_by_BD(background_df)

    t2 = time.process_time()
    # ** 效能注意：此處計算了整個 df 的 'interestYes' **
    prob_accounts_by_Yes = get_prob_accounts_by_Yes(data, threshold=threshold)
    
    t3 = time.process_time()
    # 將所有有問題的 uCode 合併 (使用 set 以提高效率)
    prob_accounts = set(prob_accounts_by_Yes).union(set(prob_accounts_by_BD))
    
    # 建立篩選條件：uCode *或* target_uCode 在黑名單中
    prob_account_filt = (data['uCode'].isin(prob_accounts) | data['target_uCode'].isin(prob_accounts))
    
    t4 = time.process_time()
    # 回傳 *不* 在黑名單中的資料
    non_prob_df = data[~prob_account_filt]
    
    t5 = time.process_time()
    print(f"Time (BD): {t2-t1:.2f}s, Time (Yes): {t3-t2:.2f}s, Time (Set+Filter): {t4-t3:.2f}s, Time (Apply): {t5-t4:.2f}s")
    return non_prob_df

def drop_null_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    移除 'target_uCode' 欄位為空值 (NaN) 的資料列。
    """
    data.dropna(subset=['target_uCode'], inplace=True)
    return data


def trim(data: pd.DataFrame, event: str, bound: float = 0.999, return_outlier=False) -> pd.DataFrame:
    """
    根據特定事件 (event) 的計數，移除極端值 (Outliers)。
    例如：移除 'interestYes' 次數排行在前 0.1% 的使用者。
    
    Args:
        data (pd.DataFrame): 原始資料 (需要有 'uCode' 和 'event' 欄位).
        event (str): 用來判斷極端值的事件名稱 (例如 'interestYes').
        bound (float, optional): 百分位數邊界. 預設為 0.999 (移除前 0.1%).
    
    Returns:
        pd.DataFrame: 已經移除極端值使用者後的 DataFrame.
    """
    # 1. 計算每個 uCode 的每種 event 次數
    df_counts = data.groupby(['uCode', 'event']).size().unstack(fill_value=0)
    
    # 2. 確保目標 event 欄位存在
    if event not in df_counts.columns:
        print(f"Error: 找不到事件 '{event}'。")
        return data
    
    # 3. 找出邊界值
    quantile_value = df_counts[event].quantile(bound)
    
    # 4. 找出未超過邊界值的 uCode
    normal_users = df_counts[df_counts[event] < quantile_value].index
    
    # 5. 過濾原始 data，只保留 "正常使用者" 的資料
    if return_outlier:
        return df_counts[df_counts[event] >= quantile_value]
    else:
        return data[data['uCode'].isin(normal_users)]
    
def trim_by_col_count(data, col, target_val, bound=0.999, return_outlier=False):
    """
    根據 'col' 欄位中 'target_val' 的計數，移除離群值。
    
    Args:
        data (pd.DataFrame): 要清理的 DataFrame
        col (str): 'event' or 'act'
        target_val (str or list): 'event' 或 'act' 的子集
    """
    # 1. 計算每個 uCode 的每種 col 值的計數
    df_counts = data[['uCode', col]].groupby(['uCode', col]).size().unstack(fill_value=0)

    # 2. 確保目標欄位存在
    if target_val not in df_counts.columns:
        print(f"Error: 找不到 {target_val}")
        return data

    # 3. 找出邊界
    filt = df_counts[target_val] < df_counts[target_val].quantile(bound)
    neg_filt = ~filt
    outliers = df_counts[neg_filt].index
    
    if return_outlier:
        return df_counts[neg_filt]
    else:
        return data[~data.uCode.isin(outliers)]

# ================================================
# 4. 繪圖 (Plotting Helpers)
# ================================================

def get_pie_chart(df, col='page'):
    """
    繪制圓餅圖，預設以 'page' 欄作為分類.

    Args:
        df: (pd.DataFrame): track_event 的資料.
        col (str, optional): 作為分類的 column. 預設為 'page'. 
    """
    
    x = df[col].value_counts()
    sizes = x / x.sum()
    labels = [f'{l} ({s:.1%})' for l, s in zip(x.index, sizes)]
    top_labels = [f'{l} ({s:.1%})' if s > 0.05 else '' for l, s in zip(x.index, sizes)]
    
    fig, ax = plt.subplots()
    ax.pie(x, labels=top_labels)
    ax.legend(labels=labels, bbox_to_anchor=(1.8,1.2))
    ax.set_title(f"Pie Chart of {col}")
    return fig, ax