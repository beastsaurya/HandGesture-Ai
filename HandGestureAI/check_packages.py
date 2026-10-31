import importlib.metadata as metadata

print("Checking installed package metadata...\n")

for package in metadata.distributions():
    try:
        _ = package.entry_points
    except Exception as error:
        print("PROBLEM PACKAGE:", package._path)
        print("ERROR:", repr(error))
        break
else:
    print("All package metadata checked successfully.")