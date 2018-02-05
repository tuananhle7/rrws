import numpy as np


def wrap_angle(angle, min_angle):
    """Transforms angle (in radians) to be in
    [min_angle, min_angle + 2 * np.pi)
    """

    return (angle - min_angle) % (2 * np.pi) + min_angle


def get_enc(position, angle, fov_angle, num_bins, other_positions):
    """Encodes sensory observations within its field of vision of a fly into a
    {0, 1} vector of size num_bins.

    input:
        position: 2d np.ndarray; position of the current fly
        angle: float; angle (in radians) of the current fly
        fov_angle: float; angle (in radians) > 0 which denotes the angle of the
            field of vision
        num_bins: number of bins in the output
        other_positions: list/np.ndarray of 2d np.ndarray of other flies'
            positions

    output: np.ndarray of size num_bins with each element being either 0 or 1
    """
    result = np.zeros([num_bins])
    for other_position in other_positions:
        relative_x, relative_y = other_position - position
        absolute_angle_of_other = np.arctan2(relative_y, relative_x)
        relative_angle_of_other = wrap_angle(
            absolute_angle_of_other - (angle - fov_angle / 2), min_angle=0
        )
        if (
            (relative_angle_of_other >= 0) and
            (relative_angle_of_other <= fov_angle)
        ):
            i = int(np.clip(np.floor(
                relative_angle_of_other / fov_angle * num_bins
            ), 0, num_bins - 1))
            result[i] = 1
    return result


def select_evaluate_scatter(inputs, indices, functions, out_dim=None):
    """Pass selected inputs through selected functions.

    input:
        inputs: input to the functions; Variable [batch_size, in_dim]
        indices: Variable [batch_size] with each elememt in {0, ..., num_functions - 1}
        functions: list of functions of length num_functions; each function takes in [-1, in_dim] and outputs [-1, out_dim]
        out_dim: output dimension of each function

    output: Variable [batch_size, out_dim] where output[b, :] = functions[indices[b]](input[b, :])
    """

    if out_dim is None:
        out_dim = functions[0](inputs[0].unsqueeze(0)).size(1)

    num_functions = len(functions)
    batch_size, in_dim = inputs.size()
    result = Variable(inputs.data.new(batch_size, out_dim))
    for f_idx, f in enumerate(functions):
        indices_for_f_idx = (indices == f_idx).nonzero().squeeze(1)
        result[indices_for_f_idx, :] = f(inputs[indices_for_f_idx, :])
    return result
