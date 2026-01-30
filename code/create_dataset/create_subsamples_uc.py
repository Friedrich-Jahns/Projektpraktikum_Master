import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm
import os
import cv2



def load_array_from_h5(path, bounds="0 -1 0 -1", return_size=False):

    try:
        bounds = np.array(bounds.split(" ")).astype(float).astype(int)
    except:
        print("Boundaries not in the right format")
    with h5py.File(path, "r") as f:
        Key = "Image" if "Image" in f.keys() else "exported_data"
        shape = f[Key].shape

        if return_size:
            return shape

        # Show Full Image
        #if isinstance(bounds,str):


        if False or all(
            [bounds[i] == [0, -1, 0, -1][i] for i in range(len(bounds))]
        ):
            bounds = [0, shape[0], 0, shape[1]]
        # Mask Dimensions don't match Image Dimensions
        if len(shape) <= 2:
            img = np.zeros((bounds[1] - bounds[0], bounds[3] - bounds[2]))
        else:
            img = np.zeros(
                (bounds[1] - bounds[0], bounds[3] - bounds[2], int(shape[2]))
            )

        f[Key].read_direct(
            img, (slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3]))
        )

    return img


def get_largest_component(data):
    print("filer")


class mask_layers:
    def __init__(self, image_path, bounds):
        self.img_path = image_path

        raw_path = Path(image_path).parent.parent

        b_folder_path = raw_path / "bgr_mask"

        self.bckg_path = (
            Path(b_folder_path)
            / f"{Path(image_path).stem}-Image_Probabilities_background{Path(image_path).suffix}"
        )

        v_folder_path = raw_path / "vessel_mask"
        self.vessel_path = (
            Path(v_folder_path)
            / f"{Path(image_path).stem}-Image_Probabilities_255_8b{Path(image_path).suffix}"
        )

        self.bounds = bounds

    def image_data(self):
        dat = load_array_from_h5(self.img_path, self.bounds)
        return dat

    def background_data(self):
        dat = load_array_from_h5(self.bckg_path, self.bounds)[:, :, 1]
        return dat

    def vessel_data(self):
        dat = load_array_from_h5(self.vessel_path, self.bounds)[:, :, 0]
        return dat

    def background_data_threshed(self):
        unthreshed = self.background_data()
        dat = np.where(unthreshed >= 100, 1, 0)
        return dat

    def only_largest_component(self):
        input = self.background_data_threshed()
        input_cc = (input > 0).astype(np.uint8)
        output = cv2.connectedComponentsWithStats(input_cc, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        area = stats[1:, cv2.CC_STAT_AREA]
        main_component_index = [i for i, x in enumerate(area) if x == np.max(area)][0]
        main_component_mask = (labels == main_component_index + 1).astype("uint8") * 255
        return main_component_mask

    def vessel_data_threshed(self):
        unthreshed = self.vessel_data()
        dat = np.where(unthreshed < 75, 0, 1)
        return dat

    def vessel_data_threshed_nobg(self):
        dat = np.multiply(self.vessel_data_threshed(), self.background_data_threshed())
        return dat

    def vessel_data_threshed_nobg_cc(self):
        dat = np.multiply(self.only_largest_component(), self.vessel_data_threshed())
        return dat


cwd = Path(os.path.dirname(os.path.abspath(__file__)))
dat_path = cwd.parent.parent / "data" / "raw" / "img"
path_list = [
   dat_path
   / "PE-2025-01953-M_00_s0060_PM_Complete_Transmittance_Stitched_Flat_v000.h5"
]

# Demo
if False:

    # obj = mask_layers(path_list[0],'700 1200 5100 5700')
    obj = mask_layers(path_list[0], [0,-1,0,-1])
    #obj = mask_layers(path_list[0], "0 -1 0 -1")
    obj.only_largest_component()
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(obj.image_data(), cmap="gray")
    plt.subplot(2, 3, 2)
    plt.imshow(obj.background_data_threshed(), cmap="gray")
    plt.subplot(2, 3, 3)
    plt.imshow(obj.vessel_data_threshed(), cmap="gray")
    plt.subplot(2, 3, 4)
    plt.imshow(obj.vessel_data_threshed_nobg(), cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(obj.only_largest_component(), cmap="gray")
    plt.subplot(2, 3, 6)
    plt.imshow(obj.vessel_data_threshed_nobg_cc(), cmap="gray")
    plt.show()
    exit()


Filepath = Path(os.path.dirname(os.path.abspath(__file__)))
data_files_path = Filepath.parent.parent / "data"
paths = []
for i, path in enumerate(os.listdir(data_files_path / "raw" / "img")):
    path = Path(path)
    if path.suffix == ".h5":
        print(f"{i} : {path}")
        paths.append(data_files_path / "raw" / "img" / path)
# print(paths[1])

# key = input('Dat. Nr.')
# file_path = Path(paths[int(key)])
# img = load_array_from_h5(file_path)
res_size = 256
step_size = res_size // 2

# print(res_size)
# input()
for img in tqdm(paths):
    # print(img)
    # print(load_array_from_h5(img,return_size=True)[0])
    # input()
    save_path = data_files_path / "training" / "val"
    save_path.mkdir(parents=True, exist_ok=True)
    shape = load_array_from_h5(img, return_size=True)
    # print(shape[0]-(shape[0]%res_size))
    # input()
    for i in tqdm(range(0, shape[0] - (res_size + shape[0] % res_size), res_size // 2)):
        for j in tqdm(
            range(0, shape[1] - (res_size + shape[0] % res_size), res_size // 2)
        ):
            # print(j)
            sub_img = mask_layers(img, f"{i} {i+res_size} {j} {j+res_size}")

            plt.imsave(
                save_path / "img" / f"{img.stem}_{i}_{j}.png",
                sub_img.image_data(),
                cmap="gray",
            )
            plt.imsave(
                save_path / "mask" / f"{img.stem}_{i}_{j}.png",
                sub_img.vessel_data_threshed_nobg(),
                cmap="gray",
            )
