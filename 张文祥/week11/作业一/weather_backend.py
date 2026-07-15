import httpx
from typing import Optional

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def geocode(city: str) -> Optional[dict]:
    """
    将城市名称解析为经纬度坐标。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含 latitude / longitude / name / country / admin1 的字典，未找到则返回 None
    """
    with httpx.Client(timeout=10.0) as client:
        # 中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
        # 而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
        # 策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
        # 且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return None

        # 在候选里优先取行政级别更高的（feature_code 含 A = 某级政府驻地），
        # 其次取有人口数据的，避免落到同名小村庄
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        return {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "name": loc.get("name", city),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
        }


def get_weather_by_coords(lat: float, lon: float, location_name: str = "") -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        lat: 纬度
        lon: 经度
        location_name: 可选的地点名称（用于显示）

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        location_str = location_name or f"{lat:.2f}°N, {lon:.2f}°E"

        lines = [
            f"【{location_str}】天气报告",
            f"坐标：{lat:.2f}°N, {lon:.2f}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )

        return "\n".join(lines)


def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报（兼容旧接口）。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    geo = geocode(city)
    if geo is None:
        return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"
    location_str = f"{geo['country']} {geo['admin1']} {geo['name']}".strip()
    return get_weather_by_coords(geo["latitude"], geo["longitude"], location_str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--coords-only", action="store_true", help="仅查经纬度，不查天气")
    args = parser.parse_args()
    if args.coords_only:
        geo = geocode(args.city)
        if geo:
            print(f"纬度: {geo['latitude']}, 经度: {geo['longitude']}")
            print(f"名称: {geo['name']}, 国家: {geo['country']}, 省份: {geo['admin1']}")
        else:
            print(f"未找到城市 '{args.city}'")
    else:
        print(get_weather(args.city))
