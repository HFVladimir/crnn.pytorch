import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import csv
import argparse


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        index = 0
        for k, v in cache.items():
            #if isinstance(v, str):
                #v = v.encode()
            txn.put(k, v)
            index += 1


def createDataset(csvPath, outputPath, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    with open(csvPath) as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            imagePath = line[0]
            label = line[1].strip().replace(' ', '')
            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                continue
            with open(imagePath, 'rb') as img:
                imageBin = img.read()
            if checkValid:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue

            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label

            if lexiconList:
                lexiconKey = 'lexicon-%09d' % cnt
                cache[lexiconKey] = ' '.join(lexiconList[0])
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
            cnt += 1
        nSamples = cnt-1
        cache['num-samples'] = str(nSamples)
        writeCache(env, cache)
        print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create lmdb dataset for crnn")
    parser.add_argument('input', help='path to csv file that contains path to picture and lable')
    parser.add_argument('--output', default='./', help='path to result data.mdb file')
    args = parser.parse_args()
    createDataset(args.input, args.output)



