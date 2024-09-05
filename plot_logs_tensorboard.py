import re
import os
import tensorflow as tf

def parse_log_file(log_file):
    # Regular expression to match the log lines containing training data
    log_pattern = re.compile(
        r'\[epoch:\s*(\d+),\s*iter:\s*(\d+),.*?lr:\((.*?)\)\].*?'
        r'l_g_pix:\s*([-\d\.e+]+)\s*l_g_percep:\s*([-\d\.e+]+)\s*l_g_gan:\s*([-\d\.e+]+)\s*'
        r'l_d_real:\s*([-\d\.e+]+)\s*out_d_real:\s*([-\d\.e+]+)\s*l_d_fake:\s*([-\d\.e+]+)\s*out_d_fake:\s*([-\d\.e+]+)'
    )

    # Initialize lists to store the parsed data
    iterations = []
    epochs = []
    lrs = []
    l_g_pix = []
    l_g_percep = []
    l_g_gan = []
    l_d_real = []
    out_d_real = []
    l_d_fake = []
    out_d_fake = []

    # Read the log file and extract the data
    with open(log_file, 'r') as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                iterations.append(int(match.group(2)))
                lrs.append(float(match.group(3)))
                l_g_pix.append(float(match.group(4)))
                l_g_percep.append(float(match.group(5)))
                l_g_gan.append(float(match.group(6)))
                l_d_real.append(float(match.group(7)))
                out_d_real.append(float(match.group(8)))
                l_d_fake.append(float(match.group(9)))
                out_d_fake.append(float(match.group(10)))

    return {
        'epochs': epochs,
        'iterations': iterations,
        'learning_rate': lrs,
        'l_g_pix': l_g_pix,
        'l_g_percep': l_g_percep,
        'l_g_gan': l_g_gan,
        'l_d_real': l_d_real,
        'out_d_real': out_d_real,
        'l_d_fake': l_d_fake,
        'out_d_fake': out_d_fake
    }

def save_to_tensorboard(summary_writer, tag, values, iterations):
    with summary_writer.as_default():
        for i, value in zip(iterations, values):
            tf.summary.scalar(tag, value, step=i)

def main(log_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each log file in the log_dir
    for log_file in os.listdir(log_dir):
        if log_file.endswith('.txt'):
            log_path = os.path.join(log_dir, log_file)
            print(f"Processing {log_path}...")

            # Parse the log file
            parsed_data = parse_log_file(log_path)

            # Create a subdirectory for each log file in the output directory
            sub_dir = os.path.join(output_dir, os.path.splitext(log_file)[0])
            os.makedirs(sub_dir, exist_ok=True)

            # Initialize a TensorFlow summary writer
            summary_writer = tf.summary.create_file_writer(sub_dir)

            # Save each metric to TensorBoard summary logs
            save_to_tensorboard(summary_writer, 'learning_rate', parsed_data['learning_rate'], parsed_data['iterations'])
            save_to_tensorboard(summary_writer, 'l_g_pix', parsed_data['l_g_pix'], parsed_data['iterations'])
            save_to_tensorboard(summary_writer, 'l_g_percep', parsed_data['l_g_percep'], parsed_data['iterations'])
            save_to_tensorboard(summary_writer, 'l_g_gan', parsed_data['l_g_gan'], parsed_data['iterations'])
            save_to_tensorboard(summary_writer, 'l_d_real', parsed_data['l_d_real'], parsed_data['iterations'])
            save_to_tensorboard(summary_writer, 'out_d_real', parsed_data['out_d_real'], parsed_data['iterations'])
            save_to_tensorboard(summary_writer, 'l_d_fake', parsed_data['l_d_fake'], parsed_data['iterations'])
            save_to_tensorboard(summary_writer, 'out_d_fake', parsed_data['out_d_fake'], parsed_data['iterations'])

            print(f"Saved TensorBoard logs to {sub_dir}")

    print("Processing complete. Use TensorBoard to visualize the results.")

# Example usage:
if __name__ == "__main__":
    log_dir = "path/to/your/log/files"  # Replace with your actual log directory
    output_dir = "path/to/output/tensorboard/logs"  # Replace with your desired output directory
    main(log_dir, output_dir)
