import collections
import re
import tensorflow as tf

init_checkpoint = "/Users/xy/Project/new_cmsc/cmsc473fall21-grover/read_checkpoint/discrimination_generator=medium_discriminator=grover_discsize=medium_dataset=p=0.96_model.ckpt-1562"

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    return (assignment_map, initialized_variable_names)

(assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint([], init_checkpoint)
for n in initialized_variable_names:
    print(n)