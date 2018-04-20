from tachyon.SPN2 import SPN
import tensorflow as tf
from timeit import default_timer as timer

from struct_gen import ConvSPN
from spn_file_conv import TachyonFileGenerator

x_size = 32
y_size = 32
x_shifts = 8
y_subdivs = 2

cv = ConvSPN(x_size, y_size, x_shifts, y_subdivs)
cv.print_stat()

file_generator = TachyonFileGenerator(cv)

model_filename = "tachyon_model.spn"
file_generator.write_to_file(model_filename)

# data_filename = "tachyon_data.txt"
# data_text = ""
# data_count = 1000
# entry_size = x_size * y_size
# for i in range(data_count):
#     entry_text = ""
#     for i in range(entry_size):
#         entry_text += "1,"
#
#     entry_text = entry_text[:-1] + "\n"
#     data_text += entry_text
#
# file = open(data_filename,'w')
# file.write(data_text)

spn = SPN()

spn.make_fast_model_from_file(model_filename, random_weights=False,cont=False)
# spn.add_data(data_filename, 'train', cont=False)
#
# spn.start_session()
#
# spn.train(1, data=spn.data.train, minibatch_size=512)
#
# compilation = timer()
#
# start = timer()
# spn.train(10, data=spn.data.train, minibatch_size=512)
# end = timer()
#
# print("Done " + str(end - start) + "s")
