import os
import site;

proj_path = os.path.abspath("../../")

d = [x for x in os.listdir(proj_path) if not x.startswith(".")]
for e in d:
    f = proj_path + "/" + e
    site_pkg = site.getsitepackages()[0] + "/" + e
    try:
        os.symlink(f, site_pkg)
    except FileExistsError:
        print("Already exists")
