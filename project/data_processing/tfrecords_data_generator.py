from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2


def parse_tfr_elem(element):
    feature_description = {
        "city_name": tf.io.FixedLenFeature([], tf.string),
        "depth_img": tf.io.FixedLenFeature([], tf.int64),
        "depth_mask": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "mask_raw": tf.io.FixedLenFeature([], tf.string),
        "width": tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(element, feature_description)

    depth_img = features["depth_img"]
    depth_mask = features["depth_mask"]
    height = features["height"]
    image_raw = features["image_raw"]
    mask_raw = features["mask_raw"]
    width = features["width"]

    img = tf.io.decode_raw(
        image_raw, tf.uint8
    )  # tf.io.parse_tensor(image_raw, out_type=tf.uint8)
    img = tf.cast(img, tf.int32)
    img_arr = tf.reshape(img, shape=[height, width, depth_img])
    img_arr = tf.cast(img_arr, tf.float32) / 255.0
    mask = tf.io.decode_raw(
        mask_raw, tf.uint8
    )  # tf.io.parse_tensor(mask_raw, out_type=tf.uint8)
    mask = tf.cast(mask, tf.int32)
    mask_arr = tf.reshape(mask, shape=[height, width, depth_mask])

    mask_new = tf.zeros((1024, 1024, 1), tf.int32)
    mask_new = tf.where([mask_arr == 3], 1, mask_new)
    mask_new = tf.where([mask_arr == 4], 2, mask_new)
    mask_new = tf.where([mask_arr == 5], 2, mask_new)
    mask_new = tf.where([mask_arr == 6], 3, mask_new)
    mask_new = tf.where([mask_arr == 7], 3, mask_new)
    mask_new = tf.where([mask_arr == 1], 4, mask_new)
    mask_new = tf.where([mask_arr == 2], 4, mask_new)
    mask_new = tf.where([mask_arr == 12], 5, mask_new)
    mask_new = tf.where([mask_arr == 13], 5, mask_new)
    mask_new = tf.where([mask_arr == 10], 6, mask_new)
    mask_new = tf.where([mask_arr == 11], 6, mask_new)
    mask_new = tf.where([mask_arr == 14], 7, mask_new)
    mask_new = tf.where([mask_arr == 15], 7, mask_new)
    mask_new = tf.where([mask_arr == 16], 8, mask_new)
    mask_new = tf.where([mask_arr == 17], 8, mask_new)
    mask_new = tf.where([mask_arr == 8], 9, mask_new)
    mask_new = tf.where([mask_arr == 9], 9, mask_new)
    mask_arr = mask_new[0]

    return (img_arr, mask_arr)


def load_tfrecords_dataset(dataset_path: str):
    dataset = tf.data.TFRecordDataset(dataset_path)
    dataset = dataset.map(parse_tfr_elem)
    return dataset


def get_good_indices(csv_path: str):
    df = pd.read_csv(csv_path)
    return df.loc[df["category"] == "good"].iloc[:, 0].tolist()


def convert_tf_records_to_png_images(dataset: tf.data.Dataset, good_img_indices: list):
    img_count = 0
    path = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset-pranet/train"
    for idx, (image, mask) in enumerate(tqdm(dataset.as_numpy_iterator())):
        if idx in good_img_indices:
            image = (image * 255).astype(np.uint8)
            cv2.imwrite(f"{path}/images/image_{idx}.png", image)
            cv2.imwrite(f"{path}/masks/image_{idx}.png", mask)
            img_count += 1

        # End streaming if all good images are processed
        if img_count == len(good_img_indices):
            break

    print(f"Total good images: {img_count}")


if __name__ == "__main__":

    quality_csv_file_path = "/home/furkan/gd/pranet/datasets/202109_nrw_cleaned/quality_reviewed/comparison_all.csv"
    indices = get_good_indices(quality_csv_file_path)
    dataset_path = "/home/furkan/gd/pranet/datasets/202109_nrw_cleaned/single_data/greenventory-NRW_train.tfrecords"
    dataset = load_tfrecords_dataset(dataset_path)
    convert_tf_records_to_png_images(dataset, good_img_indices=indices)
