# download_uci_with_mirror.py
"""
UCI 数据多镜像下载器 (解决 archive.ics.uci.edu 访问超时)
==========================================================

【问题】
直接从 archive.ics.uci.edu 下载在国内经常 timeout

【方案】
依次尝试多个镜像源：
  1. UCI 官方源（archive.ics.uci.edu）
  2. UCI 备用源（archive-beta.ics.uci.edu）  
  3. GitHub 镜像（多个团队备份在 GitHub 上）
  4. 国内学术镜像
  5. 提示用户手动下载方式

【运行】
python download_uci_with_mirror.py
"""

import os
import sys
import time
import requests
from io import StringIO
import pandas as pd

# 镜像列表（按可靠性排序）
MIRRORS = [
    {
        'name': 'UCI 官方',
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv',
        'timeout': 30,
    },
    {
        'name': 'UCI Beta',
        'url': 'https://archive-beta.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip',
        'timeout': 30,
        'is_zip': True,
    },
    {
        'name': 'GitHub 镜像 1',
        'url': 'https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/data/beijing_pm25.csv',
        'timeout': 20,
    },
    # 可继续加更多镜像...
]


def download_with_retry(url, timeout=30, max_retries=2):
    """带重试的下载"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'text/csv,application/octet-stream',
    }
    for attempt in range(max_retries):
        try:
            print(f"   尝试 {attempt + 1}/{max_retries}...")
            r = requests.get(url, headers=headers, timeout=timeout, stream=True)
            r.raise_for_status()
            
            # 显示进度（如果有 Content-Length）
            total = int(r.headers.get('Content-Length', 0))
            if total:
                print(f"   文件大小: {total/1024/1024:.2f} MB")
            
            content = b''
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                content += chunk
                downloaded += len(chunk)
                if total and downloaded % (total // 10) < 8192:
                    print(f"   进度: {downloaded/total*100:.0f}%")
            return content
        except requests.exceptions.Timeout:
            print(f"   ⏱️  超时")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return None


def try_all_mirrors():
    """依次尝试所有镜像"""
    for i, mirror in enumerate(MIRRORS, 1):
        print(f"\n[{i}/{len(MIRRORS)}] {mirror['name']}: {mirror['url'][:60]}...")
        content = download_with_retry(mirror['url'], mirror['timeout'])
        if content:
            print(f"   ✅ {mirror['name']} 下载成功！")
            
            # 处理 ZIP 格式
            if mirror.get('is_zip'):
                import zipfile
                from io import BytesIO
                try:
                    z = zipfile.ZipFile(BytesIO(content))
                    csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
                    return z.read(csv_name)
                except Exception as e:
                    print(f"   ❌ ZIP 解压失败: {e}")
                    continue
            
            return content
    return None


def show_manual_instructions():
    """显示手动下载指引"""
    print()
    print("=" * 70)
    print("所有自动下载方式都失败")
    print("=" * 70)
    print()
    print("请按以下任一方式手动获取数据：")
    print()
    print("【方式 1】从 UCI 官网手动下载")
    print("  1. 浏览器访问: https://archive.ics.uci.edu/dataset/381/")
    print("  2. 点击 'Download'")
    print("  3. 解压后把 PRSA_data_2010.1.1-2014.12.31.csv 重命名为")
    print("     beijing_pm25_raw.csv 放到当前目录")
    print()
    print("【方式 2】用同学/朋友的账号挂 VPN 后下载")
    print()
    print("【方式 3】从 Kaggle 镜像下载")
    print("  https://www.kaggle.com/datasets/uciml/pm25-data-for-five-chinese-cities")
    print()
    print("【方式 4】临时用合成数据验证流程")
    print("  python offline_data_generator.py")
    print("  ⚠️ 仅用于工程验证，正式提交不能用")
    print()


def main():
    print("=" * 70)
    print("UCI Beijing PM2.5 多镜像下载器")
    print("=" * 70)
    
    output = 'beijing_pm25_raw.csv'
    if os.path.exists(output):
        print(f"\n⚠️  {output} 已存在 ({os.path.getsize(output)/1024/1024:.2f} MB)")
        ans = input("是否重新下载？[y/N]: ").strip().lower()
        if ans != 'y':
            print("使用现有文件")
            return 0
    
    content = try_all_mirrors()
    
    if not content:
        show_manual_instructions()
        return 1
    
    # 保存
    if isinstance(content, bytes):
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin1')
    else:
        text = content
    
    with open(output, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print()
    print(f"💾 已保存: {output}")
    print(f"   大小: {os.path.getsize(output)/1024/1024:.2f} MB")
    
    # 验证
    try:
        df = pd.read_csv(output)
        print(f"\n✅ 数据验证成功")
        print(f"   行数: {len(df):,}")
        print(f"   列数: {len(df.columns)}")
        print(f"   列名: {list(df.columns)}")
    except Exception as e:
        print(f"⚠️  数据格式可能异常: {e}")
    
    print()
    print("下一步: python data_collector.py  (会自动加载这个文件做清洗)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
