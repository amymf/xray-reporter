import os 
import pandas as pd
from collections import defaultdict

images_dir = '../.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images'
projections_path = '../.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/indiana_projections.csv'
reports_path = '../.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/indiana_reports.csv'

projections_df = pd.read_csv(projections_path)
reports_df = pd.read_csv(reports_path)

# Group images by uid - some have two (frontal and lateral)
uid_to_imgs = defaultdict(list)  
for _, row in projections_df.iterrows():
    uid = row['uid']
    filename = row['filename']
    uid_to_imgs[uid].append(filename)

# Get report for each image
uid_to_report = {}
for _, row in reports_df.iterrows():
    uid = row['uid']
    uid_to_report[uid] = {
        'indication': row.get('indication', ''),
        'findings': row.get('findings', ''),
        'impression': row.get('impression', '')
    }
    
dataset = []
for uid, img_filenames in uid_to_imgs.items():
    report = uid_to_report.get(uid, {})
    dataset.append({
        'uid': uid,
        'images': img_filenames,
        'indication': report.get('indication', ''),
        'findings': report.get('findings', ''),
        'impression': report.get('impression', '')
    })

# Check 
print(dataset[0])