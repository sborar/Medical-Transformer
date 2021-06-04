import os,gzip
dataname = "mvision_train/img/image_testscan.npz"
meta_dataname = "mvision_train/img/meta_testscan.pkz"

data = np.load(dataname,allow_pickle=True)["arr_0"]
meta_data = np.load(gzip.open(meta_dataname, 'rb'),allow_pickle=True)

masksname = "mvision_train/mask_testscan"
masks = []

for filename in os.listdir(masksname):
    f = os.path.join(masksname, filename)
    # checking if it is a file
    mask = np.load(gzip.open(masksname + '/' + filename, 'rb'),allow_pickle=True)
    masks.append(mask)

shape = meta_data['shape']  # scan shape
spacing = (meta_data['ND_SliceSpacing'], meta_data['PixelSpacing'][1], meta_data['PixelSpacing'][0])

roi_data = masks[0]

shape = roi_data['shape']  # shape of scan
bbox = roi_data['bbox']
cropped_mask = roi_data['roi']

mask = np.zeros(shape, dtype=np.bool)

b = [bbox[i] for i in [0, 3, 1, 4, 2, 5]]  # get it in (z_min, z_max, y_min, y_max, x_min, x_max)
mask[b[0]: b[1], b[2]: b[3], b[4]: b[5]] = cropped_mask