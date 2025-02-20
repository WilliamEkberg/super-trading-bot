import math

def sigmoid(x: float) -> float:
    """Computes the sigmoid of x."""
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + str(err))
        return 0.0  # fallback value