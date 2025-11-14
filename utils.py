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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

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


def get_background(columns: list = None, gender: str = 'F') -> pd.DataFrame:
    """
    讀取使用者背景資料 (users_females.csv 或 users_males.csv)。
    
    Args:
        columns (list, optional): 需要讀取的欄位. 預設為 None (讀取全部).
        gender (str, optional): 'F' (女性) 或 'M' (男性). 預設為 'F'.
    
    Returns:
        pd.DataFrame: 使用者背景資料.
    """
    if gender == 'F':
        file_name = 'users_females.csv'
    else:
        file_name = 'users_males.csv'
        
    data = pd.read_csv(DATA_PATH / file_name)

    # 只保留 last_login_ymd 在202307後的帳號
    filt = data['last_login_ymd'] >= 202307

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
        background_df: users_males.csv or users_females.csv
        gender: 要加入的性別
    """
    # 建立 uCode 到 gender 的對應字典
    gender_dict = dict(zip(background_df.uCode, background_df.gender))
    data['gender'] = data['uCode'].map(gender_dict)
    return data

# ===================================
# 3. 資料清理 (Clean Data)
# ===================================

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