from mftool import Mftool
mf = Mftool()
matches = mf.get_available_schemes("HDFC")
count = 0
for code, name in matches.items():
    print(code, "-", name)
    count += 1
    if count == 10:
        break
