import os
import json
import glob
import tqdm
from argparse import ArgumentParser
import ipdb;pdb=ipdb.set_trace
import numpy as np
import random
import cv2
import shutil


obj_list = ['Custom']
categories_list = [{"id": 1, "name": "Custom", "supercategory": "None"}]
WIDTH = 1024
HEIGHT = 1024

def read_json(name):
    try:
        try:
            with open(name, 'rt') as f:
                data = eval(eval(f.read()))
        except:
            data = json.load(open(name))
    except:
        with open(name, 'rt') as f:
            data = eval(f.read())

    return data


def write_json(name, data):
    with open(name, 'wt') as f:
        f.write(json.dumps(data))


# 根据图像，找到对应的object的标注
def find_items(images, anns):
    lists = []
    for img in images:
        image_id = img['id']
        for ann in anns:
            if image_id == ann['image_id']:
                lists.append(ann)
    return lists


def write_txt(anns, labels_path):
    for img in tqdm.tqdm(anns['images']):
        img_id = img['id']
        image_name = img['file_name']
        label_txt_filename = image_name.split('.bmp')[0] + '.txt'
        label_txt_path = os.path.join(labels_path, label_txt_filename)
        if not os.path.isfile(label_txt_path):
            with open(label_txt_path, 'wt') as f:
                for ann in anns['annotations']:
                    image_id = ann['image_id']
                    if image_id == img_id:
                        x1 = ann['bbox'][0]
                        y1 = ann['bbox'][1]
                        w  = ann['bbox'][2]
                        h  = ann['bbox'][3]
                        class_id  = 0
                        x_center  = (x1 + int(w / 2)) / WIDTH
                        x_center  = round(float(x_center), 6)
                        y_center  = (y1 + int(h / 2)) / HEIGHT
                        y_center  = round(float(y_center), 6)
                        relative_width  = w / WIDTH
                        relative_width  = round(float(relative_width), 6)
                        relative_height = h / HEIGHT
                        relative_height = round(float(relative_height), 6)
                        result = str(class_id).ljust(3) + '%.06f'%x_center + ' ' + '%.06f'%y_center + ' ' + '%.06f'%relative_width + ' ' + '%.06f'%relative_height + '\n'
                        f.write(result)


def convert_coco_to_txt_per_image(train_anns, val_anns, project_path = '1600x_train'):
    labels_path = os.path.join(project_path, 'labels')
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    train_labels_path = os.path.join(labels_path, 'train')
    if not os.path.exists(train_labels_path):
        os.makedirs(train_labels_path)
    write_txt(train_anns, train_labels_path)

    val_labels_path = os.path.join(labels_path, 'val')
    if not os.path.exists(val_labels_path):
        os.makedirs(val_labels_path)
    write_txt(val_anns, val_labels_path)

    print('convert_coco_to_txt_per_image done!')


class COCOLabel:
    def __init__(self, annotations_path, display_flag=0):
        self.annotations_path = annotations_path
        self.images = []
        self.anns = []
        self.display = display_flag
        self.create_annotations()
    

    def create_annotations(self):

        #json_paths = glob.glob(self.annotations_path+'/*.json')
        json_list = [item for item in os.listdir(self.annotations_path) if '.json' in item]
        json_paths = [os.path.join(self.annotations_path, item) for item in json_list]
        #print(json_list)
        #ipdb.set_trace()
    
        object_id = 0
        image_id = 0
        for i,json_name in enumerate(json_list):
            image_id = i
            image_name = json_name.split('.json')[0] + '.bmp'
            img = {"id": image_id, "file_name": image_name}
            self.images.append(img)

            image_path = os.path.join(self.annotations_path, image_name)
            image_array = cv2.imread(image_path)

            json_path = os.path.join(self.annotations_path, json_name)
            #print("json_path = ", json_path)
            data = read_json(json_path)


            for item in data["shapes"]:
                x1, y1 = item['points'][0]
                x2, y2 = item['points'][1]
                x3, y3 = item['points'][2]
                x4, y4 = item['points'][3]
                w, h = x3-x1, y3-y1
                if (w <= 0) or (h <= 0) or (x1 != x4) or (x2 != x3) or (y1 != y2) or (y3 != y4):
                    print('json_path = %s item = %s' %(json_path, str(item)))
                    #ipdb.set_trace()
                else:
                    bbox = [x1, y1, w, h]
                    area = w*h
                    category = item['label']
                    category_id = 0
                    for item_c in categories_list:
                        if category == item_c['name']:
                            category_id = item_c['id']
                            break
                    ann = {"id": object_id, "image_id": image_id, "category_id": category_id, "iscrowd": 0, 'area': area, "bbox": bbox}
                    #print('ann = ', ann)
                    self.anns.append(ann)
                    object_id += 1

                    cv2.rectangle(image_array, (x1, y1), (x1+w, y1+h), (255, 255, 0), 2)
            cv2.imwrite(image_path.split('.bmp')[0] + '_gt.png', image_array)
            # cv2.imshow(json_name, image_array)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # ipdb.set_trace()


    def split_train_val(self, val_ratio = 0.1, images_path, project_path):
        random.seed(30)
        random.shuffle(self.images)
        val_num = int(len(self.images)*val_ratio)

        val_imgs = self.images[:val_num]
        val_anns = find_items(val_imgs, self.anns)

        train_imgs = self.images[val_num:]
        train_anns = find_items(train_imgs, self.anns)

        train_path = os.path.join(project_path, 'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        val_path = os.path.join(project_path, 'val')
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        anns_path = os.path.join(project_path, 'annotations')
        if not os.path.exists(anns_path):
            os.makedirs(anns_path)

        for img in val_imgs:
            image_name = img['file_name']
            file_image = os.path.join(images_path, image_name)
            shutil.copy(file_image, val_path)
        for img in train_imgs:
            image_name = img['file_name']
            file_image = os.path.join(images_path, image_name)
            shutil.copy(file_image, train_path)

        val_anns_path = os.path.join(anns_path, 'instances_val.json')
        val_data = {"categories": categories_list,
                    "images": val_imgs,
                    "annotations": val_anns}
        write_json(val_anns_path, val_data)

        train_anns_path = os.path.join(anns_path, 'instances_train.json')
        train_data = {"categories": categories_list,
                      "images": train_imgs,
                      "annotations": train_anns}
        write_json(train_anns_path, train_data)

        return val_imgs, val_anns, train_imgs, train_anns



if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument("-jp", "--json_path", help="json路径或者目录", default="custom")
    # args = parser.parse_args()
    # annotations_path = args.json_path
    # coco_label = COCOLabel(annotations_path)
    # val_imgs, val_anns, train_imgs, train_anns = coco_label.split_train_val(val_ratio = 0.1, images_path = annotations_path, project_path = annotations_path + '_train')
    train_anns_filename = 'custom_train/annotations/instances_train.json'
    train_anns = read_json(train_anns_filename)
    val_anns_filename = 'custom_train/annotations/instances_val.json'
    val_anns = read_json(val_anns_filename)
    convert_coco_to_txt_per_image(train_anns, val_anns, project_path = 'custom_train')
    # print(val_anns.keys())
    # print('val_anns.categories = ', val_anns['categories'])
    # ipdb.set_trace()
    # print('val_anns.images = ', val_anns['images'])
    # ipdb.set_trace()
    # print('val_anns.annotations = ', val_anns['annotations'])
    # ipdb.set_trace()