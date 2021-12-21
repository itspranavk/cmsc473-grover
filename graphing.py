import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input", None,
    "Path to .npy file containing probabilities.")

flags.DEFINE_string(
    "output", None,
    "Path to output .png file.")

def main(_):
    res2 = np.transpose(np.load(FLAGS.input))

    machine = res2[0].copy()
    human = res2[1].copy()

    x = np.array([i for i in range(len(res2[0])) ])

    machine_data = plt.scatter(x,machine, c ="blue",
        linewidths = 1,
        marker ="^", 
        s = 50,
        label = "Machine")
    human_data = plt.scatter(x, human, c ="orange", 
        linewidths = 2, 
        marker ="s", 
        s = 30)

    plt.ylabel("Probability")
    plt.xlabel("No. of sentences substituted")
    plt.title("Article 2 (Fake)")

    plt.legend((machine_data, human_data), ("Machine", "Human"))
    plt.savefig(FLAGS.output)
    plt.show()

if __name__ == "__main__":
    flags.mark_flag_as_required("input")
    flags.mark_flag_as_required("output")
    tf.app.run()