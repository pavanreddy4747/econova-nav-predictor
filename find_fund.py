from mftool import Mftool
mf = Mftool()
matches = mf.get_available_schemes("HDFC")
for code, name in matches.items():
    if "Growth" in name and "Direct" in name:
        print(code, "-", name)
