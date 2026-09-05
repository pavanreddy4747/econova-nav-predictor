from mftool import Mftool
mf = Mftool()

matches = mf.get_available_schemes("SBI")
for code, scheme_name in matches.items():
    if "Blue" in scheme_name or "Large" in scheme_name:
        print(code, "-", scheme_name)
