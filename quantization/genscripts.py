import glob

script_filename = 'processall'
file_path_list = glob.glob('../weights_data/*.txt')
file_path_list.remove(file_path_list[0])
file_names = []
for path in file_path_list:
    file_names.append(path.split('/')[-1])

NUM_FILES = len(file_path_list)

with open(script_filename, "w+") as f:
    f.write("#! /bin/bash\n\n")
    for idx in range(NUM_FILES):
        f.write("perl quantization.pl {} {}\n".format(file_path_list[idx], file_names[idx]))

    f.write("\n")
    for idx in range(NUM_FILES):
        f.write("perl dequantization.pl {} dq_{}\n".format(file_names[idx], file_names[idx]))

    f.write("\n")
    for idx in range(NUM_FILES):
        f.write("rm {}\n".format(file_names[idx]))
        f.write("mv dq_{} {}\n".format(file_names[idx], file_names[idx]))

    
