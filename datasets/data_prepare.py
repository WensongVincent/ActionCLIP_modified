import os
import random
import cv2
import glob
from tqdm import tqdm
 
from multiprocessing import Process
from multiprocessing import cpu_count
import pandas as pd
 
def parse_kinetics_annotations(input_csv, ignore_is_cc=False):
    """Returns a parsed DataFrame.
 
    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'
 
    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)
 
    return df
 
def dump_frames(video_path,img_out_path):
 
    cap = cv2.VideoCapture(video_path)
    if os.path.exists(video_path) == False:
        print('no file:',video_path)
 
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(fcount):
        try:
            ret, frame = cap.read()
            assert ret
            frame = cv2.resize(frame,(224,224))
            cv2.imwrite('%s/img_%05d.jpg' % (img_out_path, i), frame)
        except Exception as e:
            print(str(e))
            break
 
    #print('{} done'.format(video_path))
 
    return fcount
 
def process_video(data_list,label_list,input_dir,output_dir):
    for row in tqdm(data_list):
        label = row[0]
        youtube_id = row[1]
        time_start = row[2]
        time_end = row[3]
 
        video_name = '%s_%06d_%06d.mp4' % (youtube_id,time_start,time_end)
        video_path = os.path.join(input_dir,video_name)
        img_out_path = os.path.join(output_dir,label.replace(' ','_'),youtube_id)
        if os.path.exists(img_out_path) == False:
            os.makedirs(img_out_path)
 
        frame_count = dump_frames(video_path,img_out_path)
 
        # for i, l_row in label_list.iterrows():
        #     if l_row['name'] == label:
        #         str_w = img_out_path + ' ' + str(frame_count) + ' ' + str(i) + '\n'
        #         f.write(str_w)
 
    # f.close()
 
def dump_video_frames():
    input_dir = '/disk2/k400/k400_val'
    output_dir = '/disk1/k400/k400_val_img'
    input_csv = '/disk2/k400/val.csv'
    input_label_csv = 'lists/kinetics_400_labels.csv'
 
    dataset = parse_kinetics_annotations(input_csv)
    label_list = pd.read_csv(input_label_csv)
 
    data_list = dataset.values.tolist()
    # process_video(data_list, label_list,input_dir,output_dir)
 
    n_processes = cpu_count()
    processes_list = []
    random.shuffle(data_list)
 
    for n in range(n_processes):
        sub_list = data_list[
                   n * int(len(data_list) / n_processes + 1): min((n + 1) * int(len(data_list) / n_processes + 1),
                                                                  len(data_list))]
        processes_list.append(Process(target=process_video, \
                                      args=(sub_list, \
                                            label_list, \
                                            input_dir, \
                                            output_dir)))
    for p in processes_list:
        p.start()
 
def gen_label_list():
    input_dir = '/disk1/k400_img/k400_val_img'
    output_file = 'lists/k4001/k400_val_frames.txt'
    input_label_csv = 'lists/kinetics_400_labels.csv'
 
    label_list = pd.read_csv(input_label_csv)
    label_list = label_list.values.tolist()
 
    video_list = []
 
    labels = os.listdir(input_dir)
    for label in tqdm(labels):
        label_idx = -1
        for row in label_list:
            if label == row[1].replace(' ','_'):
                label_idx = row[0]
                break
        if label_idx == -1:
            print('no find label index')
            break
 
        videos = os.listdir(os.path.join(input_dir,label))
 
        for video in videos:
            files = glob.glob(os.path.join(input_dir,label,video) + '/*.jpg')
 
            if len(files) == 0:
                continue
 
            f_str = str(os.path.join(input_dir,label,video)) \
                    + ' ' + str(len(files)-1) \
                    + ' ' + str(label_idx) \
                    + '\n'
 
            video_list.append(f_str)
 
    random.shuffle(video_list)
    with open(output_file,'w') as f:
        for video in video_list:
            f.write(video)
 
if __name__ == '__main__':
 
    dump_video_frames()
    gen_label_list()
