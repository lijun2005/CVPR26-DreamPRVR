import os 
import h5py
import numpy as np 
import array 
from tqdm import tqdm 

def read_dict(filepath):
    f = open(filepath,'r')
    a = f.read()
    dict_data = eval(a)
    f.close()
    return dict_data

class BigFile:

    def __init__(self, datadir):
        self.nr_of_images, self.ndims = map(int, open(os.path.join(datadir, 'shape.txt')).readline().split())
        id_file = os.path.join(datadir, "id.txt")
        # python 3
        self.names = open(id_file, 'rb').read().strip().split()
        for i in range(len(self.names)):
            self.names[i] = str(self.names[i], encoding='ISO-8859-1')

        # python 2
        # self.names = open(id_file).read().strip().split()

        assert (len(self.names) == self.nr_of_images)
        self.name2index = dict(zip(self.names, range(self.nr_of_images)))
        self.binary_file = os.path.join(datadir, "feature.bin")
        print("[%s] %dx%d instances loaded from %s" % (self.__class__.__name__, self.nr_of_images, self.ndims, datadir))

    def read(self, requested, isname=True):
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert (min(requested) >= 0)
            assert (max(requested) < len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        if len(index_name_array) == 0:
            return [], []

        index_name_array.sort(key=lambda v: v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        vecs = [None] * nr_of_images
        offset = np.float32(1).nbytes * self.ndims

        res = array.array('f')
        fr = open(self.binary_file, 'rb')
        fr.seek(index_name_array[0][0] * offset)
        res.fromfile(fr, self.ndims)
        previous = index_name_array[0][0]

        for next in sorted_index[1:]:
            move = (next - 1 - previous) * offset
            # print next, move
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            previous = next

        fr.close()

        return [x[1] for x in index_name_array], [res[i * self.ndims:(i + 1) * self.ndims].tolist() for i in
                                                  range(nr_of_images)]

    def read_one(self, name):
        renamed, vectors = self.read([name])
        return vectors[0]

    def shape(self):
        return [self.nr_of_images, self.ndims]

def convert(visual_feat, video2frames, save_name="feature.hdf5"):
    feature_save_file = h5py.File(save_name, "w")
    video_ids = video2frames.keys()
    for video_id in tqdm(video_ids):
        frame_list = video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(visual_feat.read_one(frame_id))
        frame_vecs = np.array(frame_vecs)
        feature_save_file[video_id] = frame_vecs
    feature_save_file.close()


if __name__ == "__main__":
    # replace paths below according to downloaded ms-sl files
    # visual_feats = BigFile("activitynet/FeatureData/i3d")
    # video2frames = read_dict("activitynet/FeatureData/i3d/video2frames.txt")
    data_root = "/data2/lianniu/PRVR_data/PRVR/tvr/FeatureData/i3d_resnet"
    visual_feats = BigFile(data_root)
    video2frames = read_dict(f"{data_root}/video2frames.txt")
    convert(visual_feats, video2frames,f'{data_root}/feature.hdf5')