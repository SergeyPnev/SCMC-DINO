from defaults.bases import *
import h5py


class ImagesDS(BaseSet):
    def __init__(
        self,
        dataset_params,
        csv_file,
        img_dir,
        extend_train_set=False,
        mode="train",
        site=1,
        channels=[1, 2, 3, 4, 5, 6],
        batch_size=None,
    ):
        if mode == "eval":
            mode = "test"
        self.mode = mode
        self.attr_from_dict(dataset_params)
        self.name = self.__class__.__name__
        self.extend_train_set = extend_train_set
        df = pd.read_csv(csv_file)
        self.n_classes = 1139
        self.n_batches = len(df.experiment.unique())
        self.batch_size = batch_size
        df = df[(df.dataset == mode) & (df.cell_type == "HUVEC")]
        if (self.mode == "train") and (not self.extend_train_set):
            df = df
        elif (self.mode == "train") and (self.extend_train_set):
            df = pd.concat([df] * 20)
        self.df = df
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = self.df.shape[0]

        self.img_channels = 6
        self.img_size = 512
        self.is_multiclass = False
        self.labels_to_int = {str(x): x for x in range(self.n_classes)}
        self.int_to_labels = {int(val): key for key, val in self.labels_to_int.items()}
        self.labels_to_int_batch = {str(x): int(x[-2:]) for x in df.experiment.unique()}
        self.int_to_labels_batch = {
            val: key for key, val in self.labels_to_int_batch.items()
        }
        self.transform = Compose([Resize(512, interpolation=InterpolationMode.BICUBIC)])

    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return ToTensor()(img)

    def _get_img_path(self, index, channel):
        experiment, well, plate, site = (
            self.records[index].experiment,
            self.records[index].well,
            self.records[index].plate,
            self.records[index].site,
        )
        return "/".join(
            [
                self.img_dir,
                experiment,
                f"Plate{plate}",
                f"{well}_s{site}_w{channel}.png",
            ]
        )

    def __getitem__(self, index):
        return self.read_sample(index)

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

    def read_sample(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        X = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths]).type(
            torch.FloatTensor
        )
        X = self.transform(X)
        X = X.transpose(0, 2)
        y = int(self.records[index].sirna_id)
        key = int(self.records[index].experiment[-2:]) - 1
        plate = key * 10 + int(self.records[index].plate)
        return X, y, key, plate


data_params = {
    "data_location": "./data",
    "dataset": "RXRX1A",
    "download_data": True,
    "validation_size": 0,
}

# /PATH/TO/METDADATA/metadata.csv
csv_file = "/storage/user/datasets/RXRX1/rxrx1/metadata.csv"
img_dir = "/storage/user/datasets/RXRX1/rxrx1/images"  # /PATH/TO/IMAGES
# /PATH/TO/H5py/STORAGE/RXRX1_large_plate_info_corrected_size.hdf5
hdf5_path = (
    "/storage/user/datasets/RXRX1/rxrx1/RXRX1_large_plate_info_corrected_size.hdf5"
)


data_train = ImagesDS(
    dataset_params=data_params, mode="train", csv_file=csv_file, img_dir=img_dir
)
data_test = ImagesDS(
    dataset_params=data_params, mode="test", csv_file=csv_file, img_dir=img_dir
)

num_train_samples = len(data_train)
num_test_samples = len(data_test)

train_shape = (num_train_samples, 512, 512, 6)
test_shape = (num_test_samples, 512, 512, 6)

total_size = num_train_samples + num_test_samples
total_shape = (total_size, 512, 512, 6)

# open a hdf5 file and create earrays
f = h5py.File(hdf5_path, mode="w")

f.create_dataset("train_img", total_shape, np.uint8)
f.create_dataset("train_labels", (total_size,), np.uint16)
f.create_dataset("train_batch", (total_size,), np.uint16)
f.create_dataset("train_plate", (total_size,), np.uint16)


# loop over train paths
for i in range(len(data_train)):
    if i % 100 == 0 and i > 1:
        print("Train data: {}/{}".format(i, len(data_train)))
    item = data_train.__getitem__(i)
    sample = item[0]
    labels = item[1]
    batch = item[2]
    plate = item[3]
    img = np.array(sample)
    img = np.uint8((img * 255))
    f["train_img"][i, ...] = img
    f["train_labels"][i] = labels
    f["train_batch"][i] = batch
    f["train_plate"][i] = plate


# loop over train paths
for i in range(len(data_test)):
    if i % 1000 == 0 and i > 1:
        print("Train data: {}/{}".format(i, len(data_test)))
    item = data_test.__getitem__(i)
    sample = item[0]
    labels = item[1]
    batch = item[2]
    plate = item[3]
    img = np.array(sample)
    img = np.uint8((img * 255))
    index_ = i + num_train_samples
    f["train_img"][index_, ...] = img
    f["train_labels"][index_] = labels
    f["train_batch"][index_] = batch
    f["train_plate"][index_] = plate

f.close()
