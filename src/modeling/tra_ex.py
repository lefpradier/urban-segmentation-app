from pathlib import Path

x = "data/raw/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/train"
# print([f for f in glob.iglob(x + "**/*.png", recursive=True)])

for path in Path(x).rglob("*.png"):
    print(path)
