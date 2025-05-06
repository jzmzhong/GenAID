import os
import torchaudio
from tqdm import tqdm

DATA_DIR = "/home/s2526235/AccentedTTS/data/VCTK-Corpus-0.92-24kHz"
WAV_DIR = os.path.join(DATA_DIR, "wav48_silence_trimmed")
SPK2ACC_PATH = os.path.join(DATA_DIR, "speaker2accent_GenAIDlabels.txt")
OUT_PATH = os.path.join(DATA_DIR, "all_file_paths.csv")


def read_spk2acc(path):
    spk2acc = {}
    with open(path, encoding="utf-8", mode="r") as f:
        for i, line in enumerate(f):
            if i > 0:
                spk, acc, _ = line.split("\t")
                spk2acc[spk] = acc
    return spk2acc

spk2acc = read_spk2acc(SPK2ACC_PATH)

ID = 0
all_entries = ["ID,utt_id,wav,wav_format,text,duration,speaker,gender,accent"]

for root, dirs, files in sorted(os.walk(WAV_DIR)):
    for dir in sorted(dirs):
        full_dir = os.path.join(root, dir)
        for file in sorted(os.listdir(full_dir)):
            if file.endswith(".flac"):
                utt_id = file[:-5]
                spk = utt_id.split("_")[0]
                acc = spk2acc[spk]
                utt_id = "vctk#wav48_silence_trimmed/" + spk + "/" + file[:-5] # to match the utterance id in coqui-ai yourtts speaker/accent manager
                spk = "VCTK_" + spk
                full_path = os.path.join(full_dir, file)
                info = torchaudio.info(full_path)
                dur = info.num_frames / info.sample_rate
                full_path = full_path.replace(DATA_DIR, "$data_root")
                
                # ID,utt_id,wav,wav_format,text,duration,speaker,gender,accent
                entry = ",".join([str(ID), utt_id, full_path, "flac", "", str(dur), spk, "", acc])
                
                all_entries.append(entry)
                ID += 1

with open(OUT_PATH, 'w') as f:
    for entry in all_entries:
        f.write(entry + '\n')
