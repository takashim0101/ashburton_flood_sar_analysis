from asf_search import ASFSession, geo_search
from shapely.geometry import Point
import datetime
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
# .netrc ファイルによる認証のため、セッションを作成するのみ
session = ASFSession()

# Ashburtonの座標
ashburton_point = Point(171.75, -43.9)

# 検索条件
def search_and_download(start_date, end_date, label):
    print(f"🔍 Searching {label} images from {start_date} to {end_date}...")
    results = geo_search(
        intersectsWith=ashburton_point.buffer(0.2).wkt,
        platform=['Sentinel-1'],
        processingLevel='GRD',
        start=start_date,
        end=end_date,
        polarization='VV',
        beamMode='IW'
    )
    if results:
        print(f"✅ Found {len(results)} {label} scenes. Downloading first one...")
        results[0].download(path='data/')
    else:
        print(f"⚠️ No {label} data found.")

# 洪水前後の検索・ダウンロード
search_and_download(datetime.date(2021, 5, 1), datetime.date(2021, 5, 15), "pre-flood")
search_and_download(datetime.date(2021, 5, 20), datetime.date(2021, 6, 5), "post-flood")

