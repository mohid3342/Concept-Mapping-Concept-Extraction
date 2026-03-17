import json

def read_jsonl(file_path):
    """Reads a JSONL file and returns the data as a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    """Writes a list of dictionaries to a JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def leave_one_out_split(data):
    """Generates Leave-One-Out pairs (train, test)."""
    for i in range(len(data)):
        # Test set will be one lecture (slide)
        test_set = [data[i]]
        # Train set will be all other lectures
        train_set = data[:i] + data[i+1:]
        yield train_set, test_set

def generate_jsonl_files(input_file):
    """Generates Leave-One-Out train/test files from the input JSONL file."""
    # Read the original data
    data = read_jsonl(input_file)
    
    # Generate LOO pairs and save the train/test JSONL files
    for idx, (train_set, test_set) in enumerate(leave_one_out_split(data)):
        train_file = f'train_{idx + 1}.jsonl'
        test_file = f'test_{idx + 1}.jsonl'
        
        # Write train and test files
        write_jsonl(train_file, train_set)
        write_jsonl(test_file, test_set)
        print(f'Generated {train_file} and {test_file}')

# Example usage
input_file = 'C:\Users\Mohid\OneDrive\Documents\GitHub\Concept-Mapping-Concept-Extraction\Models\Qwen2\concept_extraction_training.jsonl'  # Replace with your JSONL file path
generate_jsonl_files(input_file)