#!/usr/bin/env python3
"""Create processed features dataset from raw E4 CSV folders.

Produces a combined CSV with sliding-window features compatible with the trained model.

Usage:
  python create_processed_dataset.py --data-root "data/Engagnition Dataset" --out data/processed_features.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import joblib
import math


def to_datetime_series(unix_series):
    # Detect units (s or ms) and convert to pandas datetime
    vals = unix_series.dropna().astype(float)
    if vals.empty:
        return pd.to_datetime([])
    median = vals.median()
    # heuristics: if > 1e12 -> microseconds? if >1e11 -> milliseconds; >1e9 -> seconds
    if median > 1e12:
        unit = 'us'
    elif median > 1e11:
        unit = 'ms'
    elif median > 1e9:
        unit = 's'
    else:
        unit = 's'
    return pd.to_datetime(unix_series.astype(float), unit=unit)


def resample_to_1hz(df, time_col='UnixTime'):
    df = df.dropna(subset=[time_col])
    if df.empty:
        return df
    times = to_datetime_series(df[time_col])
    df = df.copy()
    df.index = times
    df = df.sort_index()
    # drop duplicate timestamps (keep first)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]
    start = df.index[0].ceil('S')
    end = df.index[-1].floor('S')
    if start >= end:
        # too short
        return df.resample('1S').mean().interpolate()
    new_idx = pd.date_range(start, end, freq='1S')
    df = df.reindex(df.index.union(new_idx))
    df = df.interpolate(method='time').reindex(new_idx)
    return df


def extract_basic_features(arr):
    if len(arr) == 0:
        return dict(mean=np.nan, std=np.nan, var=np.nan, max=np.nan, min=np.nan, peak_count=0, motion_energy=0.0)
    a = np.asarray(arr, dtype=float)
    mean = float(np.nanmean(a))
    std = float(np.nanstd(a))
    var = float(np.nanvar(a))
    mx = float(np.nanmax(a))
    mn = float(np.nanmin(a))
    # peaks
    try:
        peaks, _ = find_peaks(a)
        peak_count = int(len(peaks))
    except Exception:
        peak_count = 0
    # motion energy: sum squared diffs
    if len(a) >= 2:
        me = float(np.sum(np.diff(a) ** 2))
    else:
        me = 0.0
    return dict(mean=mean, std=std, var=var, max=mx, min=mn, peak_count=peak_count, motion_energy=me)


def window_features(df1s, window_size=60, step=30):
    # df1s: DataFrame indexed by 1s datetime index, columns: GSR, Temp, Acc_X,Y,Z, Acc_SVM
    n = len(df1s)
    results = []
    if n < window_size:
        return results
    starts = range(0, n - window_size + 1, step)
    for s in starts:
        e = s + window_size
        win = df1s.iloc[s:e]
        feat = {}
        # basic signals
        signals = ['GSR', 'Temperature']
        for sig in signals:
            if sig in win.columns:
                f = extract_basic_features(win[sig].values)
            else:
                f = extract_basic_features([])
            feat.update({f"{sig}_{k}": v for k, v in f.items()})
        # accelerometer axes
        axes = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Acc_SVM']
        for ax in axes:
            if ax in win.columns:
                f = extract_basic_features(win[ax].values)
            else:
                f = extract_basic_features([])
            feat.update({f"{ax}_{k}": v for k, v in f.items()})
        results.append((win.index[0], feat))
    return results


def find_numeric_col(df, exclude=None):
    exclude = exclude or []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def process_participant(part_path):
    # look for files
    acc_file = part_path / 'E4AccData.csv'
    gsr_file = part_path / 'E4GsrData.csv'
    tmp_file = part_path / 'E4TmpData.csv'
    engagement_file = part_path / 'EngagementData.csv'
    if not (acc_file.exists() and gsr_file.exists() and tmp_file.exists()):
        return []
    try:
        acc = pd.read_csv(acc_file)
        gsr = pd.read_csv(gsr_file)
        tmp = pd.read_csv(tmp_file)
    except Exception as e:
        print(f"Failed to read CSVs for {part_path}: {e}")
        return []

    # detect columns
    # Acc expects X,Y,Z
    acc_cols = acc.columns.tolist()
    # assume UnixTime present
    for name in ['X', 'Acc_X', 'accX', 'x']:
        if name in acc_cols:
            acc_x = name
            break
    else:
        # pick first numeric col besides UnixTime
        acc_x = find_numeric_col(acc, exclude=['UnixTime'])
    # try Y and Z
    for name in ['Y', 'Acc_Y', 'accY', 'y']:
        if name in acc_cols:
            acc_y = name
            break
    else:
        acc_y = find_numeric_col(acc, exclude=['UnixTime', acc_x])
    for name in ['Z', 'Acc_Z', 'accZ', 'z']:
        if name in acc_cols:
            acc_z = name
            break
    else:
        acc_z = find_numeric_col(acc, exclude=['UnixTime', acc_x, acc_y])

    # GSR/Temp cols
    gsr_col = find_numeric_col(gsr, exclude=['UnixTime'])
    tmp_col = find_numeric_col(tmp, exclude=['UnixTime'])

    # convert and resample
    acc1 = resample_to_1hz(acc, time_col='UnixTime')
    gsr1 = resample_to_1hz(gsr, time_col='UnixTime')
    tmp1 = resample_to_1hz(tmp, time_col='UnixTime')

    # build unified DF
    df = pd.DataFrame(index=acc1.index)
    # acc axes
    if acc_x in acc1.columns:
        df['Acc_X'] = acc1[acc_x]
    if acc_y in acc1.columns:
        df['Acc_Y'] = acc1[acc_y]
    if acc_z in acc1.columns:
        df['Acc_Z'] = acc1[acc_z]
    # compute SVM
    if 'Acc_X' in df.columns and 'Acc_Y' in df.columns and 'Acc_Z' in df.columns:
        df['Acc_SVM'] = np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)

    if gsr_col and gsr1 is not None and gsr_col in gsr1.columns:
        # align by index: use intersection
        df = df.join(gsr1[[gsr_col]], how='left')
        df = df.rename(columns={gsr_col: 'GSR'})
    if tmp_col and tmp1 is not None and tmp_col in tmp1.columns:
        df = df.join(tmp1[[tmp_col]], how='left')
        df = df.rename(columns={tmp_col: 'Temperature'})

    # drop rows where all signals are NaN
    df = df.dropna(how='all')
    if df.empty:
        return []

    # extract windows
    feats = window_features(df, window_size=60, step=30)

    # try to read label if exists
    label_map = None
    label_series = None
    if engagement_file.exists():
        try:
            edf = pd.read_csv(engagement_file)
            # look for a time column to align, else use a static label
            # find label col
            label_cols = [c for c in edf.columns if 'engag' in c.lower() or 'label' in c.lower() or 'class' in c.lower()]
            if label_cols:
                label_series = edf[label_cols[0]]
        except Exception:
            label_series = None

    out = []
    for start_time, feat in feats:
        row = {k: v for k, v in feat.items()}
        row['participant'] = part_path.name
        row['start_time'] = start_time.isoformat()
        # label not aligned here; leave as NaN unless mapping available
        row['label'] = np.nan
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data/Engagnition Dataset', help='Root folder containing condition subfolders')
    p.add_argument('--out', default='data/processed_features.csv')
    args = p.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print('Data root not found:', root)
        return

    rows = []
    for condition in root.iterdir():
        if not condition.is_dir():
            continue
        for part in condition.iterdir():
            if not part.is_dir():
                continue
            print('Processing', condition.name, part.name)
            out = process_participant(part)
            for r in out:
                r['condition'] = condition.name
            rows.extend(out)

    if not rows:
        print('No features extracted.')
        return

    df = pd.DataFrame(rows)

    # reorder columns: participant, condition, start_time, label, then feature columns
    # attempt to use models/feature_columns.joblib for feature order
    feature_cols = []
    model_cols_path = Path('models/feature_columns.joblib')
    if model_cols_path.exists():
        try:
            feature_cols = joblib.load(model_cols_path)
        except Exception:
            feature_cols = [c for c in df.columns if c not in ['participant','condition','start_time','label']]
    else:
        feature_cols = [c for c in df.columns if c not in ['participant','condition','start_time','label']]

    cols = ['participant', 'condition', 'start_time', 'label'] + [c for c in feature_cols if c in df.columns]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print('Wrote', out_path, 'rows=', len(df))


if __name__ == '__main__':
    main()
