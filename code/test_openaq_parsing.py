# -*- coding: utf-8 -*-
"""
test_openaq_parsing.py
======================
对 OpenAQv3Collector 的解析逻辑做 mock 测试, 不需要真实网络。
验证 _parse_record 与 extract_pm25_sensor_ids 是否能处理:
  1. v3 真实 API 响应结构
  2. 老 v2 字段（向后兼容）
  3. 各种异常输入
"""

import sys
import os

# ✅ v3.6: Windows GBK 兼容性
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _windows_compat import configure_utf8_stdout
    configure_utf8_stdout()
except ImportError:
    pass

from real_multi_city_collector import OpenAQv3Collector, OPENAQ_PARAM_IDS


def test_extract_pm25_sensor_ids():
    """测试从 location 中提取 PM2.5 sensor"""
    print("=" * 70)
    print("测试 1: extract_pm25_sensor_ids")
    print("=" * 70)

    # 案例 1: 标准 v3 location 含多个 sensor
    location_v3 = {
        'id': 12345,
        'name': '北京示范站',
        'sensors': [
            {'id': 100, 'parameter': {'id': 2, 'name': 'pm25', 'units': 'µg/m³'}},
            {'id': 101, 'parameter': {'id': 1, 'name': 'pm10'}},
            {'id': 102, 'parameter': {'id': 5, 'name': 'no2'}},
            {'id': 103, 'parameter': {'name': 'pm25'}},  # 缺 id 也要识别
        ],
    }
    sensor_ids = OpenAQv3Collector.extract_pm25_sensor_ids(
        OpenAQv3Collector.__new__(OpenAQv3Collector), location_v3
    )
    expected = [100, 103]
    assert sensor_ids == expected, f"FAIL: got {sensor_ids}, expected {expected}"
    print(f"  ✅ v3 标准结构: 提取到 sensors {sensor_ids}")

    # 案例 2: 没有 sensors 字段
    location_empty = {'id': 99, 'name': 'empty'}
    assert OpenAQv3Collector.extract_pm25_sensor_ids(
        OpenAQv3Collector.__new__(OpenAQv3Collector), location_empty
    ) == [], "FAIL: empty location should return []"
    print("  ✅ 无 sensors 字段: 返回空列表")

    # 案例 3: sensors 是 None
    location_none = {'id': 99, 'sensors': None}
    assert OpenAQv3Collector.extract_pm25_sensor_ids(
        OpenAQv3Collector.__new__(OpenAQv3Collector), location_none
    ) == [], "FAIL: sensors=None should return []"
    print("  ✅ sensors=None: 返回空列表")

    # 案例 4: parameter 字段是 None
    location_param_none = {
        'id': 99,
        'sensors': [{'id': 1, 'parameter': None}],
    }
    assert OpenAQv3Collector.extract_pm25_sensor_ids(
        OpenAQv3Collector.__new__(OpenAQv3Collector), location_param_none
    ) == [], "FAIL: parameter=None should not crash"
    print("  ✅ parameter=None: 不崩溃, 返回空列表")


def test_parse_record():
    """测试 measurement 响应解析"""
    print("\n" + "=" * 70)
    print("测试 2: _parse_record")
    print("=" * 70)

    location = {
        'id': 12345,
        'name': '北京海淀示范站',
        'coordinates': {'latitude': 39.96, 'longitude': 116.30},
    }

    # 案例 1: v3 标准响应（period.datetimeFrom.utc）
    m_v3 = {
        'value': 78.5,
        'period': {
            'datetimeFrom': {'utc': '2024-01-15T08:00:00Z'},
            'datetimeTo':   {'utc': '2024-01-15T09:00:00Z'},
        },
        'parameter': {'id': 2, 'name': 'pm25'},
    }
    rec = OpenAQv3Collector._parse_record(m_v3, '北京', location, sensor_id=100)
    assert rec is not None
    assert rec['pm25'] == 78.5
    assert rec['city'] == '北京'
    assert rec['source'] == 'OpenAQ_v3'
    assert rec['timestamp'] == '2024-01-15T08:00:00Z'
    print(f"  ✅ v3 标准响应 (period.datetimeFrom.utc): "
          f"pm25={rec['pm25']}, ts={rec['timestamp']}")

    # 案例 2: v2 兼容响应（date.utc）
    m_v2 = {
        'value': 45.0,
        'date': {'utc': '2024-01-15T10:00:00Z', 'local': '2024-01-15T18:00:00+08:00'},
    }
    rec = OpenAQv3Collector._parse_record(m_v2, '上海', location, 200)
    assert rec is not None
    assert rec['pm25'] == 45.0
    print(f"  ✅ v2 兼容响应 (date.utc): pm25={rec['pm25']}")

    # 案例 3: 测量值缺失
    m_no_value = {'period': {'datetimeFrom': {'utc': '2024-01-01T00:00:00Z'}}}
    rec = OpenAQv3Collector._parse_record(m_no_value, '广州', location, 300)
    assert rec is None
    print("  ✅ 无 value: 返回 None")

    # 案例 4: 测量值是负数（无效）
    m_neg = {'value': -5.0, 'period': {'datetimeFrom': {'utc': '2024-01-01T00:00:00Z'}}}
    rec = OpenAQv3Collector._parse_record(m_neg, '深圳', location, 400)
    assert rec is None
    print("  ✅ 负数 value: 返回 None")

    # 案例 5: value 是字符串 (有些代理 API 会序列化错)
    m_str = {'value': '99.5', 'period': {'datetimeFrom': {'utc': '2024-01-01T00:00:00Z'}}}
    rec = OpenAQv3Collector._parse_record(m_str, '武汉', location, 500)
    assert rec is not None
    assert rec['pm25'] == 99.5
    print(f"  ✅ value 是字符串: 自动转换 pm25={rec['pm25']}")

    # 案例 6: 时间字段缺失
    m_no_time = {'value': 50.0}
    rec = OpenAQv3Collector._parse_record(m_no_time, '成都', location, 600)
    assert rec is None
    print("  ✅ 无时间字段: 返回 None")

    # 案例 7: coordinates 缺失
    location_no_coords = {'id': 999, 'name': 'no_coords'}
    m = {'value': 30.0, 'period': {'datetimeFrom': {'utc': '2024-01-01T00:00:00Z'}}}
    rec = OpenAQv3Collector._parse_record(m, '西安', location_no_coords, 700)
    assert rec is not None
    assert rec['latitude'] is None
    assert rec['longitude'] is None
    print("  ✅ 无 coordinates: latitude/longitude=None, 不崩溃")


def test_api_endpoint_format():
    """验证 URL 与参数格式符合 OpenAQ v3 文档"""
    print("\n" + "=" * 70)
    print("测试 3: API 端点格式 (静态检查)")
    print("=" * 70)

    assert OpenAQv3Collector.BASE == "https://api.openaq.org/v3", \
        "FAIL: BASE 应为 v3 端点"
    print(f"  ✅ BASE = {OpenAQv3Collector.BASE}")

    assert OPENAQ_PARAM_IDS['pm25'] == 2, "FAIL: PM2.5 ID 应为 2"
    print(f"  ✅ pm25 parameter_id = {OPENAQ_PARAM_IDS['pm25']}")

    cities = OpenAQv3Collector.CHINA_CITIES
    assert len(cities) >= 8, "FAIL: 至少 8 个城市"
    for city, info in cities.items():
        assert 'lat' in info and 'lon' in info and 'radius_km' in info
    print(f"  ✅ 中国城市覆盖: {len(cities)} 个")


def main():
    print("\n" + "=" * 70)
    print("OpenAQ v3 解析逻辑 mock 测试")
    print("=" * 70 + "\n")

    test_extract_pm25_sensor_ids()
    test_parse_record()
    test_api_endpoint_format()

    print("\n" + "=" * 70)
    print("✅ 所有 mock 测试通过")
    print("=" * 70)
    print("\n下一步: 在用户本机配置 OPENAQ_API_KEY 后运行")
    print("  python real_multi_city_collector.py --dry-run --api-key $OPENAQ_API_KEY")


if __name__ == '__main__':
    main()
