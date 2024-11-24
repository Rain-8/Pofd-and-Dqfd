import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

def map_scores(dqfd_scores, ddqn_scores, xlabel='Episodes', ylabel='Scores'):
    """
    Plots the mean scores of DDQN and DQfD for comparison.

    Parameters:
        dqfd_scores (list or array): Mean scores for DQfD.
        ddqn_scores (list or array): Mean scores for DDQN.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(12, 6))
    
    # Check if scores are not empty before plotting
    if dqfd_scores is not None and len(dqfd_scores) > 0:
        plt.plot(dqfd_scores, label='DQfD', color='red', linestyle='--', linewidth=2)
    if ddqn_scores is not None and len(ddqn_scores) > 0:
        plt.plot(ddqn_scores, label='DDQN', color='blue', linestyle='-', linewidth=2)
    
    # Adding labels, legend, and grid
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Performance Comparison: DDQN vs DQfD')
    plt.legend()
    plt.grid()
    plt.show()


# Define the path to the DQfD model checkpoint
checkpoint_path = "C:\\Users\\SowmyaG\\projects\\cartpole\\DQfD\\model/DQfD_model"

# Load the TensorFlow model (assuming TensorFlow 1.x)
try:
    with tf.compat.v1.Session() as sess:
        # Load the meta graph and restore the weights
        saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + ".meta")
        saver.restore(sess, checkpoint_path)

        # Access the graph
        graph = tf.compat.v1.get_default_graph()

        # Print all available tensor names to help find the correct one
        print("Available tensors in the graph:")
        for op in graph.get_operations():
            print(op.name)

        # Example: Access a tensor if needed (update the tensor name based on the printed list)
        # q_values_tensor = graph.get_tensor_by_name("q_values:0")
        print("Model loaded successfully from:", checkpoint_path)

except Exception as e:
    print(f"Error loading model: {e}")

# Load the mean scores for DDQN and DQfD
try:
    with open('ddqn_mean_scores.p', 'rb') as f:
        ddqn_scores = pickle.load(f)

    with open('dqfd.p', 'rb') as f:
        dqfd_scores = pickle.load(f)

    # Print a sample of the scores for verification
    print("DDQN Scores Sample:", ddqn_scores[:5])
    print("DQfD Scores Sample:", dqfd_scores[:5])

except Exception as e:
    print(f"Error loading score data: {e}")

# Plot the results for DDQN and DQfD
try:
    print("Plotting results...")
    map_scores(dqfd_scores=dqfd_scores, ddqn_scores=ddqn_scores,
               xlabel='Episodes', ylabel='Scores')
    print("Plotting completed successfully.")
except Exception as e:
    print(f"Error during plotting: {e}")

