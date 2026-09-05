from mftool import Mftool
mf = Mftool()

codes = ["108467", "120585", "108466", "108465", "120586"]
for code in codes:
    data = mf.get_scheme_historical_nav(code)
    if data and "data" in data and len(data["data"]) > 0:
        latest = data["data"][0]
        scheme_name = data.get("scheme_name", "Unknown")
        print(f"{code} - {scheme_name} - Latest NAV: {latest['nav']} on {latest['date']}")
    else:
        print(f"{code} - No data found")
