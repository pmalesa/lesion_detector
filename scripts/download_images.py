import os
import shutil
import urllib.request
import zipfile

import pandas as pd

# URLs for the zip files
urls = [
    "https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip",
    "https://nihcc.box.com/shared/static/l9e1ys5e48qq8s409ua3uv6uwuko0y5c.zip",
    "https://nihcc.box.com/shared/static/48jotosvbrw0rlke4u88tzadmabcp72r.zip",
    "https://nihcc.box.com/shared/static/xa3rjr6nzej6yfgzj9z6hf97ljpq1wkm.zip",
    "https://nihcc.box.com/shared/static/58ix4lxaadjxvjzq4am5ehpzhdvzl7os.zip",
    "https://nihcc.box.com/shared/static/cfouy1al16n0linxqt504n3macomhdj8.zip",
    "https://nihcc.box.com/shared/static/z84jjstqfrhhlr7jikwsvcdutl7jnk78.zip",
    "https://nihcc.box.com/shared/static/6viu9bqirhjjz34xhd1nttcqurez8654.zip",
    "https://nihcc.box.com/shared/static/9ii2xb6z7869khz9xxrwcx1393a05610.zip",
    "https://nihcc.box.com/shared/static/2c7y53eees3a3vdls5preayjaf0mc3bn.zip",
    "https://nihcc.box.com/shared/static/2zsqpzru46wsp0f99eaag5yiad42iezz.zip",
    "https://nihcc.box.com/shared/static/8v8kfhgyngceiu6cr4sq1o8yftu8162m.zip",
    "https://nihcc.box.com/shared/static/jl8ic5cq84e1ijy6z8h52mhnzfqj36q6.zip",
    "https://nihcc.box.com/shared/static/un990ghdh14hp0k7zm8m4qkqrbc0qfu5.zip",
    "https://nihcc.box.com/shared/static/kxvbvri827o1ssl7l4ji1fngfe0pbt4p.zip",
    "https://nihcc.box.com/shared/static/h1jhw1bee3c08pgk537j02q6ue2brxmb.zip",
    "https://nihcc.box.com/shared/static/78hamrdfzjzevrxqfr95h1jqzdqndi19.zip",
    "https://nihcc.box.com/shared/static/kca6qlkgejyxtsgjgvyoku3z745wbgkc.zip",
    "https://nihcc.box.com/shared/static/e8yrtq31g0d8yhjrl6kjplffbsxoc5aw.zip",
    "https://nihcc.box.com/shared/static/vomu8feie1qembrsfy2yaq36cimvymj8.zip",
    "https://nihcc.box.com/shared/static/ecwyyx47p2jd621wt5c5tc92dselz9nx.zip",
    "https://nihcc.box.com/shared/static/fbnafa8rj00y0b5tq05wld0vbgvxnbpe.zip",
    "https://nihcc.box.com/shared/static/50v75duviqrhaj1h7a1v3gm6iv9d58en.zip",
    "https://nihcc.box.com/shared/static/oylbi4bmcnr2o65id2v9rfnqp16l3hp0.zip",
    "https://nihcc.box.com/shared/static/mw15sn09vriv3f1lrlnh3plz7pxt4hoo.zip",
    "https://nihcc.box.com/shared/static/zi68hd5o6dajgimnw5fiu7sh63kah5sd.zip",
    "https://nihcc.box.com/shared/static/3yiszde3vlklv4xoj1m7k0syqo3yy5ec.zip",
    "https://nihcc.box.com/shared/static/w2v86eshepbix9u3813m70d8zqe735xq.zip",
    "https://nihcc.box.com/shared/static/0cf5w11yvecfq34sd09qol5atzk1a4ql.zip",
    "https://nihcc.box.com/shared/static/275en88yybbvzf7hhsbl6d7kghfxfshi.zip",
    "https://nihcc.box.com/shared/static/l52tpmmkgjlfa065ow8czhivhu5vx27n.zip",
    "https://nihcc.box.com/shared/static/p89awvi7nj0yov1l2o9hzi5l3q183lqe.zip",
    "https://nihcc.box.com/shared/static/or9m7tqbrayvtuppsm4epwsl9rog94o8.zip",
    "https://nihcc.box.com/shared/static/vuac680472w3r7i859b0ng7fcxf71wev.zip",
    "https://nihcc.box.com/shared/static/pllix2czjvoykgbd8syzq9gq5wkofps6.zip",
    "https://nihcc.box.com/shared/static/2dn2kipkkya5zuusll4jlyil3cqzboyk.zip",
    "https://nihcc.box.com/shared/static/peva7rpx9lww6zgpd0n8olpo3b2n05ft.zip",
    "https://nihcc.box.com/shared/static/2fda8akx3r3mhkts4v6mg3si7dipr7rg.zip",
    "https://nihcc.box.com/shared/static/ijd3kwljgpgynfwj0vhj5j5aurzjpwxp.zip",
    "https://nihcc.box.com/shared/static/nc6rwjixplkc5cx983mng9mwe99j8oa2.zip",
    "https://nihcc.box.com/shared/static/rhnfkwctdcb6y92gn7u98pept6qjfaud.zip",
    "https://nihcc.box.com/shared/static/7315e79xqm72osa4869oqkb2o0wayz6k.zip",
    "https://nihcc.box.com/shared/static/4nbwf4j9ejhm2ozv8mz3x9jcji6knhhk.zip",
    "https://nihcc.box.com/shared/static/1lhhx2uc7w14bt70de0bzcja199k62vn.zip",
    "https://nihcc.box.com/shared/static/guho09wmfnlpmg64npz78m4jg5oxqnbo.zip",
    "https://nihcc.box.com/shared/static/epu016ga5dh01s9ynlbioyjbi2dua02x.zip",
    "https://nihcc.box.com/shared/static/b4ebv95vpr55jqghf6bthg92vktocdkg.zip",
    "https://nihcc.box.com/shared/static/byl9pk2y727wpvk0pju4ls4oomz9du6t.zip",
    "https://nihcc.box.com/shared/static/kisfbpualo24dhby243nuyfr8bszkqg1.zip",
    "https://nihcc.box.com/shared/static/rs1s5ouk4l3icu1n6vyf63r2uhmnv6wz.zip",
    "https://nihcc.box.com/shared/static/7tvrneuqt4eq4q1d7lj0fnafn15hu9oj.zip",
    "https://nihcc.box.com/shared/static/gjo530t0dgeci3hizcfdvubr2n3mzmtu.zip",
    "https://nihcc.box.com/shared/static/7x4pvrdu0lhazj83sdee7nr0zj0s1t0v.zip",
    "https://nihcc.box.com/shared/static/z7s2zzdtxe696rlo16cqf5pxahpl8dup.zip",
    "https://nihcc.box.com/shared/static/shr998yp51gf2y5jj7jqxz2ht8lcbril.zip",
    "https://nihcc.box.com/shared/static/kqg4peb9j53ljhrxe3l3zrj4ac6xogif.zip",
]


def download_file(url: str, download_path: str, file_name: str):
    """
    Downloads a file under given URL to the specified path
    with a given file name.
    """

    with urllib.request.urlopen(url) as response:
        with open(f"{download_path}/{file_name}", "wb") as out_file:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)


def extract_key_slices(file_path: str, extract_path: str, key_slice_names: set[str]):
    """
    Extracts key slice images from the given zip file to the
    specified location.
    """

    with zipfile.ZipFile(file_path, "r") as zip_file:
        for member in zip_file.namelist():
            if member.startswith("Images_png/") and not member.endswith("/"):
                relative_path = member[len("Images_png/") :]
                parts = relative_path.split("/")

                if len(parts) != 2:
                    continue

                subfolder_id, image_name = parts
                new_filename = f"{subfolder_id}_{image_name}"

                if new_filename not in key_slice_names:
                    continue

                target_path = os.path.join(extract_path, new_filename)

                with zip_file.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)


def get_key_slice_names() -> set[str]:
    """
    Reads the DeepLesion metadata files and returns
    the set of all image names.
    """

    path = "../data/deeplesion_metadata.csv"
    metadata = pd.read_csv(path)
    image_names = set()

    for i in range(len(metadata)):
        if metadata["File_name"][i] not in image_names:
            image_names.add(metadata["File_name"][i])

    return image_names


def check_key_slices(path: str, key_slice_names: set[str]) -> bool:
    """
    Checks whether all key slices are present
    in the specified location.
    """

    extracted_key_slices = set()
    image_extensions = ".png"

    for file_name in os.listdir(path):
        if file_name.lower().endswith(image_extensions):
            print(f"Extracted file - {file_name}")
            extracted_key_slices.add(file_name)

    for key_slice in key_slice_names:
        if key_slice not in extracted_key_slices:
            return False

    return len(key_slice_names) == len(extracted_key_slices)


def download_images():
    """
    Downloads all zip files one at the time and
    extracts the key slice images from them.
    """

    download_path = "../data/deeplesion"
    target_path = "../data/deeplesion/key_slices"
    os.makedirs(target_path, exist_ok=True)
    key_slice_names = get_key_slice_names()

    for i, url in enumerate(urls):
        file_name = f"Images_png_{i + 1:02d}.zip"
        zip_path = os.path.join(download_path, file_name)
        print(f"Downloading {file_name} ...")
        download_file(url, download_path, file_name)
        print("Download successful.")
        print("Extracting images to temporary directory ...")
        extract_key_slices(zip_path, target_path, key_slice_names)
        print("All images extracted.")
        os.remove(zip_path)

    if not check_key_slices(target_path, key_slice_names):
        print("WARNING! Not all key slice images were extracted!")
    print("All key slices were downloaded successfully!")


if __name__ == "__main__":
    download_images()
