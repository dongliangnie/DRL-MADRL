# utils/data_loader.py
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def load_gps_csv(path, time_key='timestamp', lat_key='lat', lon_key='lon', id_key='vehicle_id'):
    """
    读取轨迹 CSV，必须包含：vehicle_id, timestamp, lat, lon
    返回 pandas.DataFrame，timestamp 应为 pd.Timestamp 或 int 秒。
    """
    df = pd.read_csv(path)
    # 尝试解析 timestamp
    if not np.issubdtype(df[time_key].dtype, np.datetime64):
        try:
            df[time_key] = pd.to_datetime(df[time_key])
        except Exception:
            # 如果是 unix 秒（int）
            df[time_key] = pd.to_datetime(df[time_key], unit='s')
    return df

def trajectories_to_time_series(df, start_time=None, end_time=None, slot_seconds=20):
    """
    将原始轨迹转为按 time-slot 的字典：
      { slot_idx: { vehicle_id: (lat, lon), ... }, ... }
    简化策略：在每个 slot 取最近时间点的位置（或线性插值可后续加入）。
    """
    df = df.sort_values('timestamp')
    if start_time is None:
        start_time = df['timestamp'].min()
    if end_time is None:
        end_time = df['timestamp'].max()

    start_ts = pd.to_datetime(start_time)
    end_ts = pd.to_datetime(end_time)

    total_seconds = int((end_ts - start_ts).total_seconds())
    n_slots = total_seconds // slot_seconds + 1

    # group by vehicle
    vehicles = {vid: g.sort_values('timestamp') for vid, g in df.groupby('vehicle_id')}

    result = {}
    for slot_idx in range(n_slots):
        t = start_ts + pd.Timedelta(seconds=slot_seconds * slot_idx)
        slot_dict = {}
        for vid, g in vehicles.items():
            # find the last record <= t
            g_before = g[g['timestamp'] <= t]
            if not g_before.empty:
                row = g_before.iloc[-1]
                slot_dict[vid] = (float(row['lat']), float(row['lon']))
            else:
                # no data yet -> skip or nan
                pass
        result[slot_idx] = slot_dict
    return result

def generate_pois_from_trajectories(df, n_pois=100, method='kmeans'):
    """
    根据轨迹点生成 PoI（任务点）。默认 KMeans 聚类得到热点中心作为 PoI。
    返回 numpy array shape (n_pois, 2) 用 lat/lon。
    """
    coords = df[['lat', 'lon']].dropna().values
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_pois, random_state=0).fit(coords)
        centers = kmeans.cluster_centers_
        return centers
    else:
        # 随机采样
        idx = np.random.choice(len(coords), size=n_pois, replace=False)
        return coords[idx]

def build_task_list_from_pois(pois, start_slot, n_slots, surveillance_prob=0.02, emergency_prob=0.001):
    """
    生成基于 Pois 的任务时间表：每个 slot 可能触发部分 surveillance / occasional emergency
    返回: list of tasks per slot: tasks[slot] = [ {task_id, type, location, aoi_threshold, ...}, ... ]
    """
    tasks_by_slot = [[] for _ in range(n_slots)]
    task_id = 0
    for slot in range(n_slots):
        for p in pois:
            if np.random.rand() < surveillance_prob:
                tasks_by_slot[slot].append({
                    'task_id': task_id,
                    'type': 'surveillance',
                    'location': (float(p[0]), float(p[1])),
                    'aoi_threshold': 35,
                    'data_size': 10.0
                })
                task_id += 1
            if np.random.rand() < emergency_prob:
                tasks_by_slot[slot].append({
                    'task_id': task_id,
                    'type': 'emergency',
                    'location': (float(p[0]), float(p[1])),
                    'aoi_threshold': 20,
                    'data_size': 50.0,
                    'area_size': 100.0
                })
                task_id += 1
    return tasks_by_slot
