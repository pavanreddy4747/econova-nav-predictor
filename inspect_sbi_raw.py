from mftool import Mftool
mf = Mftool()

matches = mf.get_available_schemes("SBI")
print(f"Total SBI matches: {len(matches)}")
count = 0
for code, scheme_name in matches.items():
    print(code, "-", scheme_name)
    count += 1
    if count == 20:
        break
