import csv
import random
from rnn import RNN,Net
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import tensorflow as tf
import pretty_midi
import pathlib
import glob

with open(os.path.dirname(os.path.realpath(__file__))+'/loss_data/loss.pickle', 'wb') as f:
    pass
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.00001
split = True
n_split = 4
language = 'both'
batch_size = 2
epoch = 5000
note_weight=3

def load_train_sample(language='both',split = True,n_split = 4):
    train_samples = []
    total_time_interval = 0
    total_note = 0
    if language == 'korean' or language == "both":
        for i in range(1,51):
            if i < 10:
                i = '0'+str(i)
            with open("./CSD/korean/csv/kr0"+str(i)+"b.csv",'r') as f:
                print("./CSD/korean/csv/kr0"+str(i)+"b.csv is opened!")
                reader = csv.reader(f)
                sample = []
                for j,line in enumerate(reader):
                    if j == 0:
                        continue
                    time_interval = float(line[1])-float(line[0])
                    note = int(line[2])
                    total_time_interval += time_interval
                    total_note += note
                    sample.append([time_interval,note])
                train_samples.append(sample)
    if language == 'english' or language == "both":
        for i in range(1,51):
            if i < 10:
                i = '0'+str(i)
            with open("./CSD/english/csv/en0"+str(i)+"b.csv",'r') as f:
                print("./CSD/english/csv/en0"+str(i)+"b.csv is opened!")
                reader = csv.reader(f)
                sample = []
                for j,line in enumerate(reader):
                    if j == 0:
                        continue
                    time_interval = float(line[1])-float(line[0])
                    note = int(line[2])
                    total_time_interval += time_interval
                    total_note += note
                    sample.append([time_interval,note])
                    
                train_samples.append(sample)
    print(len(train_samples),"melodies are loaded!")
    if split:
        splited_part = []
        unit_number = n_split
        for sample in train_samples:
            i = 0
            while True:
                if len(sample)-(unit_number) == i:
                    break
                splited_part.append([sample[i:i+unit_number],sample[i+unit_number]])
                i += 1
        random.shuffle(splited_part)
        print(len(splited_part),"split melodies are shuffled!")
        note_number = len(splited_part)-n_split*len(train_samples)
        avg_time_inteval = total_time_interval/note_number
        avg_note = total_note/note_number
        print('================================================')
        return splited_part,avg_time_inteval,avg_note
    else:
        print('================================================')
        return train_samples,0.44281,67.42

def load_train_sample_tf(split=True,n_split = 4):
    total_time_interval = 0
    total_note = 0
    data_dir = pathlib.Path('data/maestro-v2.0.0')
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )
    filenames = glob.glob(str(data_dir/'**/*.mid*'))
    # print('Number of files:', len(filenames))
    train_samples = []
    j = 1
    for filename in filenames:
        print("loading sample:",str(j)+'/'+str(len(filenames)))
        sample_file = filename
        pm = pretty_midi.PrettyMIDI(sample_file)
        instrument = pm.instruments[0]
        sample = []
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start
        for i, note in enumerate(sorted_notes):
            # start = note.start
            # end = note.end
            step = note.start-prev_start
            duration = note.end - note.start
            total_time_interval += duration
            total_note += note.pitch
            sample.append([note.pitch,step,duration])
            prev_start = note.start
        train_samples.append(sample)
        j += 1
        break
    if split:
        splited_part = []
        unit_number = n_split
        j = 1
        for sample in train_samples:
            i = 0
            print("spliting sample:",str(j)+'/'+str(len(filenames)))
            while True:
                if len(sample)-(unit_number) == i:
                    break
                splited_part.append([sample[i:i+unit_number],sample[i+unit_number]])
                i += 1
            j += 1
        random.shuffle(splited_part)
        print(len(splited_part),"split melodies are shuffled!")
        note_number = len(splited_part)-n_split*len(train_samples)
        avg_time_inteval = total_time_interval/note_number
        avg_note = total_note/note_number
        print('================================================')
        return splited_part,avg_time_inteval,avg_note
    else:
        print('================================================')
        return train_samples,0.44281,67.42
    
def load_train_sample_train(split=True,n_split = 4):
    total_time_interval = 0
    total_note = 0
    data_dir = pathlib.Path('train')
    filenames = glob.glob(str(data_dir/'*.mid'))
    train_samples = []
    j = 1
    for filename in filenames:
        print("loading sample:",str(j)+'/'+str(len(filenames)))
        sample_file = filename
        pm = pretty_midi.PrettyMIDI(sample_file)
        instrument = pm.instruments[0]
        sample = []
        for i, note in enumerate(instrument.notes):
            duration = note.end - note.start
            total_time_interval += duration
            total_note += note.pitch
            sample.append([note.pitch,duration])
        train_samples.append(sample)
        j += 1
    if split:
        splited_part = []
        unit_number = n_split
        j = 1
        for sample in train_samples:
            i = 0
            print("spliting sample:",str(j)+'/'+str(len(filenames)))
            while True:
                if len(sample) < unit_number+1:
                    break
                if len(sample)-(unit_number) == i:
                    break
                splited_part.append([sample[i:i+unit_number],sample[i+unit_number]])
                i += 1
            j += 1
        random.shuffle(splited_part)
        print(len(splited_part),"split melodies are shuffled!")
        note_number = len(splited_part)-n_split*len(train_samples)
        avg_time_inteval = total_time_interval/note_number
        avg_note = total_note/note_number
        print('================================================')
        return splited_part,avg_time_inteval,avg_note
    else:
        print('================================================')
        return train_samples,0.44281,67.42
    
def sampling_mini_batch(samples):
    mini_batch = random.sample(samples,batch_size)
    input_mini_batch =[]
    target_mini_batch = []
    for i in mini_batch:
        target_mini_batch.append(i[1])
        input_mini_batch.append(i[0])
    input_mini_batch = torch.Tensor(np.array(input_mini_batch)).to(device)
    target_mini_batch = torch.Tensor(np.array(target_mini_batch)).to(device)
    return input_mini_batch,target_mini_batch
    
# samples1,avg_time,avg_note = load_train_sample(language=language,split=split,n_split=n_split)
# samples,avg_time,avg_note = load_train_sample_train(split=split,n_split=n_split)
samples,avg_time,avg_note = load_train_sample_tf(split=split,n_split=n_split)
norm = torch.Tensor(np.array([avg_time,(1/note_weight)*avg_note])).to(device)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for _ in range(epoch):
    input_mini_batch,target_mini_batch = sampling_mini_batch(samples)
    y_pred = model(input_mini_batch,target_mini_batch)
    loss = ((y_pred-target_mini_batch).squeeze()**2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('===================================')
    print("epoch:"+str(_+1))
    print("loss:"+str(loss.item()))
    with open(os.path.dirname(os.path.realpath(__file__))+'/loss_data/loss.pickle', 'ab') as f:
        pickle.dump(loss.item(),f)