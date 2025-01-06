import numpy as np

# ----------------------
# Starter Code for Students
# ----------------------

def initialize_parameters(input_size, output_size):
    """Initialize weight parameters with small random values."""
    return np.random.randn(output_size, input_size) * 0.1

# Task 1: Implement the Softmax Function
# The softmax function converts a vector of values to a probability distribution.
# Each element is transformed using the exponential function, making them positive,
# and then normalized so that the sum of the resulting values is 1.
# Implement the softmax function that takes a numpy array `x` as input and returns
# a numpy array of the same shape, where each element is the softmax of `x`.
def softmax(x):
    # YOUR CODE HERE
    
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# Task 2: Implement the Scaled Dot-Product Attention Mechanism
# The attention function computes a weighted sum of values V, where the weight assigned
# to each value is computed by a compatibility function of the query Q with the corresponding key K.
# Implement the function `scaled_dot_product_attention(Q, K, V)` that computes the attention
# mechanism's output and the attention weights.
# Hint: Use the softmax function you implemented earlier for computing the attention weights.
def scaled_dot_product_attention(Q, K, V):
    # YOUR CODE HERE
     # Ensure Q, K, V have a batch dimension if they don't
    if Q.ndim == 2:
        Q = Q[np.newaxis, ...]
        K = K[np.newaxis, ...]
        V = V[np.newaxis, ...]
    
    d = Q.shape[-1]
    attention = np.matmul(Q, K.transpose(0,2,1)) / np.sqrt(d)
    attention_weights = softmax(attention)
    output = np.matmul(attention_weights, V)
    if output.shape[0] == 1:
        output = output.squeeze(0)
        attention_weights = attention_weights.squeeze(0)
    return output, attention_weights

# Task 3: Implement the Transformer Decoder Layer
# A transformer decoder layer consists of a self-attention mechanism, cross-attention with
# respect to the encoder outputs, and a position-wise feed-forward network.
# Using the `initialize_parameters` function for initializing weights, implement the transformer
# decoder layer function `transformer_decoder_layer(Q, K, V, memory, params, mask=None)`.
# Use the attention mechanism you defined in Task 2 for both self-attention and cross-attention.
# Hint: The decoder outputs should pass through a layer normalization step at the end.
def transformer_decoder_layer(Q, K, V, memory, params, mask=None):
    # YOUR CODE HERE
    Q_sa = np.dot(Q, params['W_q'])
    K_sa = np.dot(K, params['W_k'])
    V_sa = np.dot(V, params['W_v'])
    
    # Self-attention and projection, followed by layer normalization
    sa_output, _ = scaled_dot_product_attention(Q_sa, K_sa, V_sa)
    sa_output = layer_norm(np.dot(sa_output, params['W_o']) + Q)
    
    # Apply linear transformations for cross-attention using encoder output (memory)
    Q_ca = sa_output  # Output of self-attention serves as Q for cross-attention
    K_ca = np.dot(memory, params['W_m_k'])
    V_ca = np.dot(memory, params['W_m_v'])
    
    # Cross-attention and projection, followed by layer normalization
    ca_output, _ = scaled_dot_product_attention(Q_ca, K_ca, V_ca)
    ca_output = layer_norm(np.dot(ca_output, params['W_o']) + sa_output)
    
    # Position-wise feed-forward network
    ff_output = np.dot(ca_output, params['W_ff1']) + params['b_ff1']
    ff_output = np.maximum(0, ff_output)  # ReLU activation
    ff_output = np.dot(ff_output, params['W_ff2']) + params['b_ff2']
    
    # Final layer normalization
    decoder_output = layer_norm(ff_output + ca_output)
    
    return decoder_output
    
#Layer_norm is given as:
def layer_norm(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-6)
# ----------------------
# Parameters Initialization (Provided)
# ----------------------

d_model = 10  # Embedding size
d_ff = 20  # Size of the feed-forward network
vocab_size = 10  # Assuming a vocab size of 10 for simplicity

# Initialize weights (This part is provided to students)
params = {
    'W_q': initialize_parameters(d_model, d_model),
    'W_k': initialize_parameters(d_model, d_model),
    'W_v': initialize_parameters(d_model, d_model),
    'W_o': initialize_parameters(d_model, d_model),
    'W_m_k': initialize_parameters(d_model, d_model),
    'W_m_v': initialize_parameters(d_model, d_model),
    'W_ff1': initialize_parameters(d_ff, d_model),
    'b_ff1': np.zeros(d_ff),
    'W_ff2': initialize_parameters(d_model, d_ff),
    'b_ff2': np.zeros(d_model),
    'd_model': d_model
}

# Test Check 1: Softmax Function
def check_softmax():
    print("Checking the softmax function...")
    test_input = np.array([1.0, 2.0, 3.0])
    output = softmax(test_input)
    if np.allclose(output, np.array([0.09003057, 0.24472847, 0.66524096])):
        print("Softmax function seems to be implemented correctly.")
    else:
        print("Softmax function may be incorrect. Please check your implementation.")

# Test Check 2: Scaled Dot-Product Attention
def check_attention():
    print("Checking the attention mechanism...")
    Q = np.array([[1, 0, 0], [0, 1, 0]])
    K = V = np.array([[1, 2, 3], [4, 5, 6]])
    output, _ = scaled_dot_product_attention(Q, K, V)
    expected_output = np.array([[3.54902366, 4.54902366, 5.54902366], [3.54902366, 4.54902366, 5.54902366]])
    if np.allclose(output, expected_output):
        print("Attention mechanism seems to be implemented correctly.")
    else:
        print("Attention mechanism may be incorrect. Please check your implementation.")

# Test Check 3: Transformer Decoder Layer Functionality
def check_decoder_layer():
    print("Checking the transformer decoder layer...")
    Q = K = V = memory = np.random.randn(1, 10, d_model)
    output = transformer_decoder_layer(Q, K, V, memory, params)
    # Instead of just checking the shape, let's ensure there's a non-zero variance
    # across the output, indicating that the layer has applied some transformation.
    if output.shape == (1, 10, d_model) and np.var(output) != 0:
        print("Transformer decoder layer output shape is correct and shows variance across outputs.")
    else:
        print("There might be an issue with the transformer decoder layer. Please check your implementation.")

# Uncomment to run checks
check_softmax()
check_attention()
check_decoder_layer()