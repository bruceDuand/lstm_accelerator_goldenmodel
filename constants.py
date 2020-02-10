import socket

host = socket.gethostname()
CLASSES = []
NUM_MFCC = 40
NUM_FRAMES = 87
DURATION = 2  # in seconds
GENDER_CLASSES = ['M', 'F']
PICKLE_FILE_PREFIX = 'LibriSpeech-mfcc-'

CLASSES = []
MAX_CLASSES = 10

PROJECT_ROOT = '/Users/DD/Developer/lstm_gender_classifier/'
# DATASET_STR = 'dev-clean'
DATASET_STR = 'train-clean-100'
DATA_ROOT = '/Users/DD/Developer/lstm_gender_classifier/LibriSpeech/'
DATA_DIR = DATA_ROOT + DATASET_STR + '/'
SPEAKER_FILE = DATA_ROOT + 'SPEAKERS.TXT'
SPEAKER_IDX = 7
CHAPTER_IDX = 8
FILENAME_IDX = 9
NUM_CLASSES = 40

