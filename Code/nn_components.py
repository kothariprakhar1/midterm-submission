import numpy as np

def relu(x):
    """
    Classic ReLU. Keeps positive numbers, kills negatives to 0.
    """
    return np.maximum(0, x)

def softmax(logits):
    """
    Squashes raw model outputs into probabilities that add up to 1.
    """
    # Subtracting the max prevents nasty numerical overflow errors in exp
    exp_logits = np.exp(logits - np.max(logits)) 
    return exp_logits / np.sum(exp_logits)

def cross_entropy_loss(predictions, targets):
    """
    Calculates the penalty based on how far off the predictions are from the true labels.
    """
    # Clip values so we never accidentally try to take the log of 0
    predictions = np.clip(predictions, 1e-15, 1.0 - 1e-15) 
    return -np.sum(targets * np.log(predictions))

# --- Simple 2D Max Pooling Test ---
def max_pooling_2d(image_matrix):
    """
    A quick manual mock-up of 2x2 Max Pooling with a stride of 2.
    Shrinks a 4x4 matrix down to a 2x2 matrix.
    """
    h, w = image_matrix.shape
    pooled_output = np.zeros((h // 2, w // 2))
    
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            # Grab the 2x2 section
            window = image_matrix[i:i+2, j:j+2]
            # Grab the max value in that pool and save it
            pooled_output[i//2, j//2] = np.max(window)
            
    return pooled_output

if __name__ == "__main__":
    print("--- Quick Check on My Math Components ---")
    
    # Testing ReLU behavior
    test_inputs = np.array([-5.2, 0.0, 3.8])
    print(f"Inputs: {test_inputs} | After ReLU: {relu(test_inputs)}")
    
    # Testing our manual pooling logic on a fake 4x4 image patch
    fake_edge_map = np.array([
        [1, 2, 0, 3],
        [4, 8, 1, 0],
        [2, 0, 5, 7],
        [1, 3, 2, 4]
    ])
    print("\nOriginal 4x4 Map:\n", fake_edge_map)
    print("\nAfter 2x2 Max Pooling:\n", max_pooling_2d(fake_edge_map))
