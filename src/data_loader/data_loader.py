import glob


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(images_path, masks_path, n=2160):
        """Loads dataset from path"""
        imgfiles = [f for f in glob.glob(images_path)]
        print(images_path)
        print(imgfiles)
        mskfiles = [f for f in glob.glob(masks_path)]
        imgfiles.sort()
        mskfiles.sort()

        n_pic = len(imgfiles) // n

        whole_imgl = [{} for _ in range(n_pic)]
        whole_mskl = [{} for _ in range(n_pic)]
        for r, i in enumerate(zip(imgfiles, mskfiles)):
            whole_imgl[r % n_pic][int(i[0].split('/')[-1].split('_')[0])] = (i[0])
            whole_mskl[r % n_pic][int(i[0].split('/')[-1].split('_')[0])] = (i[1])
        w_imgl = [[x.get(i) for i in range(1, len(x) + 1, 1)] for x in whole_imgl]
        w_mskl = [[x.get(i) for i in range(1, len(x) + 1, 1)] for x in whole_mskl]
        return [w_imgl, w_mskl]
