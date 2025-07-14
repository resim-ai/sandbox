import csv
from pathlib import Path

def main():
    input_path = "/tmp/resim/inputs/detections.csv"
    output_path = "/tmp/resim/outputs/detections.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_path, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            # Replace path in filename column (index 0)
            row[0] = row[0].replace("/tmp/resim/inputs/", "/tmp/resim/inputs/experience/")
            writer.writerow(row)

    print(f"Updated CSV written to {output_path}")

if __name__ == "__main__":
    main()