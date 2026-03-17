import json
import os


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def leave_one_out_split(data):
    for i in range(len(data)):
        test_set = [data[i]]
        train_set = data[:i] + data[i + 1:]
        yield train_set, test_set


def generate_jsonl_files(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    data = read_jsonl(input_file)

    for idx, (train_set, test_set) in enumerate(leave_one_out_split(data)):
        train_file = os.path.join(output_dir, f'train_{idx + 1}.jsonl')
        test_file = os.path.join(output_dir, f'test_{idx + 1}.jsonl')

        write_jsonl(train_file, train_set)
        write_jsonl(test_file, test_set)

        print(f'Generated {train_file} and {test_file}')


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ✅ FIXED: file is in same directory
    input_file = os.path.join(base_dir, 'concept_extraction_training.jsonl')

    output_dir = os.path.join(base_dir, 'loo_splits')

    generate_jsonl_files(input_file, output_dir)