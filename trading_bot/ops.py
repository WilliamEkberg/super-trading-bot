import math
import torch

def sigmoid(x: float) -> float:
    """Computes the sigmoid of x."""
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + str(err))
        return 0.0  # fallback value

def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t as a PyTorch tensor.
    
    Args:
        data (list or array-like): Sequence of data points.
        t (int): Current time index.
        n_days (int): Number of days to consider for the state.
        
    Returns:
        torch.Tensor: A tensor of shape (1, n_days - 1) containing sigmoid differences 
                      between consecutive data points. If not enough data points exist, 
                      the state is padded with the first element.
    """
    d = t - n_days + 1
    if d >= 0:
        block = data[d: t + 1]
    else:
        block = [-d * [data[0]] + data[0: t + 1]]
        # Alternatively, you can pad with the first element:
        block = [data[0]] * (-d) + data[0: t + 1]

    res = [sigmoid(block[i + 1] - block[i]) for i in range(n_days - 1)]
    # Return the state as a torch tensor with shape (1, n_days-1)
    return torch.tensor([res], dtype=torch.float)
