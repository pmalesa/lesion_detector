# import pandas as pd

# path = "../data/deeplesion_metadata.csv"
# metadata = pd.read_csv(path)
# image_names = {}

# for i in range(len(metadata)):
#     if metadata["File_name"][i] in image_names:
#         image_names[metadata["File_name"][i]] += 1
#     else:
#         image_names[metadata["File_name"][i]] = 1

# # for i, image_name in enumerate(image_names):
# #     if image_names[metadata["File_name"][i]] > 1:
# #         print(f"[{i + 1}] {image_name} -> {image_names[metadata["File_name"][i]]} duplicates")

# image_metadata = metadata.loc[metadata["File_name"] == "000016_01_01_030.png"]

# print(image_metadata)

# # WIP ...
