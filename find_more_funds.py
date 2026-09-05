from mftool import Mftool
mf = Mftool()

for amc in ["ICICI Prudential", "SBI"]:
    matches = mf.get_available_schemes(amc)
    print(f"\n{amc} -- total matches: {len(matches)}")
    count = 0
    for code, scheme_name in matches.items():
        if "Growth" in scheme_name and "Direct" in scheme_name and ("Large Cap" in scheme_name or "Bluechip" in scheme_name or "Blue Chip" in scheme_name):
            print(code, "-", scheme_name)
            count += 1
    print(f"Large-cap style matches: {count}")
